import os, logging, csv, numpy as np
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
#from torch.utils.tensorboard import SummaryWriter # Keep this import if cfg.tensorboard is True, otherwise remove (based on final decision)
import time
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, load_checkpoint_inv, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, AverageMeter 
from torch.utils.data import DataLoader 
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.models import build_model_from_cfg
from torchmetrics import MeanSquaredError, MeanAbsoluteError 
from create_event_dataset import EventPointCloudDataset 
from openpoints.models.regression.reg_head import RegressionHead 
from RegressionModelWrapper import RegressionModelWrapper 
from ClassificationModelWrapper import ClassificationModelWrapper # adds this to the registry of models


#Regression-specific helper functions
def print_regression_results(loss, mse, mae, epoch_val, cfg): 
    s = f'\nE@{epoch_val}\tLoss: {loss:.4f}\tMSE: {mse:.4f}\tMAE: {mae:.4f}\n'
    logging.info(s)


def main(gpu, cfg, profile=False):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                 init_method=cfg.dist_url,
                                 world_size=cfg.world_size,
                                 rank=cfg.rank)
        dist.barrier()
    
    #Logger setup
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    
   
    if cfg.rank == 0 : 
        if cfg.common.tensorboard:
            writer = SummaryWriter(log_dir=cfg.run_dir) 
        else:
            writer = None #Set writer to None if TensorBoard is disabled
    else:
        writer = None 

    set_random_seed(cfg.common.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    logging.info(f"\n--- GPU Status Check ---")
    logging.info(f"CUDA available (PyTorch check): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"CUDA current device (cfg.rank): {cfg.rank}")
        logging.info(f"CUDA device name (device 0): {torch.cuda.get_device_name(0)}")
        logging.info(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    logging.info(f"--- End GPU Status Check ---\n")


    model = build_model_from_cfg(cfg.model).to(cfg.rank)

    logging.info(f"Model moved to device: {next(model.parameters()).device}") 
    if torch.cuda.is_available():
        logging.info(f"CUDA memory after model move: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")


    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    
    criterion = nn.BCELoss(reduction=cfg.loss.get('reduction', 'mean')).cuda()
    # criterion = nn.MSELoss(reduction=cfg.loss.get('reduction', 'mean')).cuda()

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    #optimizer = build_optimizer_from_cfg(model, lr=cfg.optimizer.lr, **cfg.optimizer)
    optimizer = build_optimizer_from_cfg(model, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    full_dataset = EventPointCloudDataset(
        h5_file_path=cfg.dataset.common.h5_file_path,
        num_points=cfg.dataset.common.num_points,
        use_transforms=cfg.dataset.common.get('transforms'), #Apply common transforms to the base dataset
        max_samples_per_category=cfg.dataset.common.get('max_samples') #Pass the max_samples from YAML here
    )
    total_samples = len(full_dataset)
    train_size=int(cfg.dataset.common.train_split*total_samples)
    val_size=int(cfg.dataset.common.val_split*total_samples)
    test_size=int(cfg.dataset.common.test_split*total_samples)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset_split, val_dataset_split, test_dataset_split = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )


    train_loader = DataLoader(
        train_dataset_split,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.get('num_workers', 4), 
        shuffle=True, 
        drop_last=True
        #,worker_init_fn=EventPointCloudDataset.worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset_split,
        batch_size=cfg.dataset.val.get('batch_size', cfg.train.batch_size), 
        num_workers=cfg.dataset.val.get('num_workers', 4),
        shuffle=False, 
        drop_last=False
        #,worker_init_fn=EventPointCloudDataset.worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset_split,
        batch_size=cfg.get('test_batch_size', cfg.dataset.val.get('batch_size', cfg.train.batch_size)),
        num_workers=cfg.get('test_num_workers', 4),
        shuffle=False, 
        drop_last=False
        #,worker_init_fn=EventPointCloudDataset.worker_init_fn
    )

    logging.info(f"Length of training dataset: {len(train_loader.dataset)}")
    logging.info(f"Length of validation dataset: {len(val_loader.dataset)}")
    logging.info(f"Length of test dataset: {len(test_loader.dataset)}")
    
    cfg.num_outputs = 1 
    validate_fn = validate_regression # TODO: Create BSE validation function

    best_val_mse = float('inf') # TODO
    best_epoch = 0

    model.zero_grad() 
    for epoch in range(cfg.train.start_epoch, cfg.epochs + 1):
        epoch_start_time = time.time()
        logging.info(f"DEBUG TIMING: Epoch {epoch} starts at {time.strftime('%H:%M:%S', time.localtime(epoch_start_time))}")

        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch) 
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1 
        
        train_loop_start = time.time()
        train_loss, train_mse, train_mae = \
            train_one_epoch(model, train_loader, optimizer, scheduler, epoch, criterion, cfg)
        train_loop_end = time.time()
        logging.info(f"DEBUG TIMING: Epoch {epoch} - train_one_epoch duration: {train_loop_end - train_loop_start:.2f} seconds")

        val_check_start = time.time()
        val_loss, val_mse, val_mae = float('inf'), float('inf'), float('inf') 
        is_best = False
        if epoch % cfg.train.val_freq == 0:
            val_loss, val_mse, val_mae = validate_fn(
                model, val_loader, criterion, cfg) 
            
            is_best = val_mse < best_val_mse 
            if is_best:
                best_val_mse = val_mse
                best_epoch = epoch
                logging.info(f'Found a better checkpoint @E{epoch} with MSE: {best_val_mse:.4f}')
            
            print_regression_results(val_loss, val_mse, val_mae, epoch, cfg)
        val_check_end = time.time()
        logging.info(f"DEBUG TIMING: Epoch {epoch} - Validation check block duration: {val_check_end - val_check_start:.2f} seconds")


        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_loss {train_loss:.4f}, val_mse {val_mse:.4f}, best val mse {best_val_mse:.4f}')
        
        tb_log_start = time.time()
        if writer is not None: 
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/mse', train_mse, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/mse', val_mse, epoch)
            writer.add_scalar('val/mae', val_mae, epoch)
            writer.add_scalar('best_val_mse', best_val_mse, epoch)
            writer.add_scalar('epoch', epoch, epoch)
        tb_log_end = time.time()
        logging.info(f"DEBUG TIMING: Epoch {epoch} - TensorBoard logging duration: {tb_log_end - tb_log_start:.2f} seconds")

        sched_step_start = time.time()
        if cfg.train.sched_on_epoch:
            scheduler.step(epoch) 

        sched_step_end = time.time()
        logging.info(f"DEBUG TIMING: Epoch {epoch} - Scheduler step duration: {sched_step_end - sched_step_start:.2f} seconds")
        
        save_ckpt_start = time.time()
        if cfg.rank == 0: 
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val_mse': best_val_mse}, 
                            is_best=is_best
                            )
        save_ckpt_end = time.time()
        logging.info(f"DEBUG TIMING: Epoch {epoch} - Checkpoint save duration: {save_ckpt_end - save_ckpt_start:.2f} seconds")

        epoch_end_time = time.time()
        logging.info(f"DEBUG TIMING: Epoch {epoch} - TOTAL epoch wall clock time: {epoch_end_time - epoch_start_time:.2f} seconds")
        #Sum of component
        logging.info(f"DEBUG TIMING: Epoch {epoch} - Sum of (Train+Val+TB+Sched+Save) durations: { (train_loop_end - train_loop_start) + (val_check_end - val_check_start) + (tb_log_end - tb_log_start) + (sched_step_end - sched_step_start) + (save_ckpt_end - save_ckpt_start):.2f} seconds")

    
    # Test the last epoch model 
    test_loss_last, test_mse_last, test_mae_last = validate_and_save_predictions(model, test_loader, criterion, cfg, split_name="last_epoch_test")
    print_regression_results(test_loss_last, test_mse_last, test_mae_last, cfg.epochs, cfg)
    if writer is not None:
        writer.add_scalar('test/loss_last_epoch', test_loss_last, cfg.epochs)
        writer.add_scalar('test/mse_last_epoch', test_mse_last, cfg.epochs)
        writer.add_scalar('test/mae_last_epoch', test_mae_last, cfg.epochs)

    # Test the best validataion model
    best_epoch_loaded, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_loss_best, test_mse_best, test_mae_best = validate_and_save_predictions(model, test_loader, criterion, cfg, split_name="best_epoch_test")
    if writer is not None:
        writer.add_scalar('test/loss_best_val_epoch', test_loss_best, best_epoch_loaded)
        writer.add_scalar('test/mse_best_val_epoch', test_mse_best, best_epoch_loaded)
        writer.add_scalar('test/mae_best_val_epoch', test_mae_best, best_epoch_loaded)
    print_regression_results(test_loss_best, test_mse_best, test_mae_best, best_epoch_loaded, cfg)

    #Close TensorBoard writer (if active)
    if writer is not None:
        writer.close() 
    
    
    if cfg.distributed: 
        dist.destroy_process_group() 

#Modified train_one_epoch function
def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, criterion, cfg): # Added criterion param
    loss_meter = AverageMeter()
    mse_meter = MeanSquaredError().to(cfg.rank)
    mae_meter = MeanAbsoluteError().to(cfg.rank)

    # pbar_val_setup_start_time = time.time()
    # pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Evaluating validation")
    # pbar_val_setup_end_time = time.time()
    # logging.info(f"DEBUG TIMING: Val - pbar setup duration: {pbar_val_setup_end_time - pbar_val_setup_start_time:.4f}s")


    model.train()  #set model to training mode
    pbar = tqdm(enumerate(train_loader), total=len(train_loader)) 
    num_iter = 0
    for idx, (data_dict, target) in pbar: 
        batch_start_time = time.time()

        data_load_start = time.time()
        data_dict['pos'] = data_dict['pos'].cuda(non_blocking=True)
        data_dict['x'] = data_dict['x'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_load_end = time.time()
        
        # if torch.cuda.is_available() and idx % cfg.train.print_freq == 0: 
        #     logging.info(f"Batch {idx} CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        #     logging.info(f"Batch {idx} CUDA memory cached: {torch.cuda.memory_cached() / (1024**2):.2f} MB")


        num_iter += 1
        
        model_compute_start = time.time()

        # print(f"pos: {np.shape(data_dict['pos'])}, x: {np.shape(data_dict['x'])}, target: {np.shape(target)}")
        logits = model(data_dict) 

        # print(f'logits: {np.shape(logits)}, target: {np.shape(target)}')
        # print(f'{logits}')
        # print(f'{target}')

        loss = criterion(logits.squeeze(-1), target.squeeze(-1)) 

        loss.backward()

        if num_iter == cfg.train.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.train.sched_on_epoch:
                scheduler.step(epoch)
        model_compute_end = time.time()


        # if idx % cfg.train.print_freq == 0:
        #     logging.info(f"DEBUG TIMING: Train Batch {idx} - Data Load+Transfer: {data_load_end - data_load_start:.4f}s")
        #     logging.info(f"DEBUG TIMING: Train Batch {idx} - Model Compute (Fwd+Bwd+Opt): {model_compute_end - model_compute_start:.4f}s")
        #     logging.info(f"DEBUG TIMING: Train Batch {idx} - Total Batch Wall Time: {time.time() - batch_start_time:.4f}s")
        #     #Check GPU Memory per batch (might spams so can remove)
        #     if torch.cuda.is_available():
        #         logging.info(f"DEBUG: Batch {idx} CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        #         logging.info(f"DEBUG: Batch {idx} CUDA memory cached: {torch.cuda.memory_cached() / (1024**2):.2f} MB")

        mse_meter.update(logits.squeeze(-1), target.squeeze(-1))
        mae_meter.update(logits.squeeze(-1), target.squeeze(-1))
        
        loss_meter.update(loss.item())
        if idx % cfg.train.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Avg {loss_meter.avg:.3f}")
    
    train_mse = mse_meter.compute()
    train_mae = mae_meter.compute()
    
    return loss_meter.avg, train_mse, train_mae


@torch.no_grad()
def validate_regression(model, val_loader, criterion, cfg): 
    model.eval() 
    loss_meter = AverageMeter()
    mse_meter = MeanSquaredError().to(cfg.rank)
    mae_meter = MeanAbsoluteError().to(cfg.rank)

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for idx, (data_dict, target) in pbar: 

        batch_val_start_time = time.time()


        data_load_val_start = time.time()
        data_dict['pos'] = data_dict['pos'].cuda(non_blocking=True)
        data_dict['x'] = data_dict['x'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_load_val_end = time.time()


        model_compute_val_start = time.time()
        logits = model(data_dict)

        # print(logits.argmax(dim=1))

        loss = criterion(logits.squeeze(-1), target.squeeze(-1))
        model_compute_val_end = time.time()

        metrics_update_val_start = time.time()
        mse_meter.update(logits.squeeze(-1), target.squeeze(-1))
        mae_meter.update(logits.squeeze(-1), target.squeeze(-1))
        loss_meter.update(loss.item())
        metrics_update_val_end = time.time()


        # if idx % cfg.train.print_freq == 0: 
        #     logging.info(f"DEBUG TIMING: Val Batch {idx} - Data Load+Transfer: {data_load_val_end - data_load_val_start:.4f}s")
        #     logging.info(f"DEBUG TIMING: Val Batch {idx} - Model Compute (Fwd+Loss): {model_compute_val_end - model_compute_val_start:.4f}s")
        #     logging.info(f"DEBUG TIMING: Val Batch {idx} - Metrics Update: {metrics_update_val_end - metrics_update_val_start:.4f}s")
        #     logging.info(f"DEBUG TIMING: Val Batch {idx} - Total Batch Wall Time: {time.time() - batch_val_start_time:.4f}s")

        if idx % cfg.train.print_freq == 0:
            pbar.set_description(f"Val Epoch [{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} MSE {mse_meter.compute():.4f} MAE {mae_meter.compute():.4f}")
    
    val_mse = mse_meter.compute()
    val_mae = mae_meter.compute()
    val_loss = loss_meter.avg 

    return val_loss, val_mse, val_mae

@torch.no_grad()
def validate_and_save_predictions(model, data_loader, criterion, cfg, split_name="test"):
    #Validates the model and saves all individual predictions and ground truth targets to disk.
    model.eval() 
    loss_meter = AverageMeter()
    mse_meter = MeanSquaredError().to(cfg.rank)
    mae_meter = MeanAbsoluteError().to(cfg.rank)

    #Lists to store predictions and targets from all batches
    all_predictions = []
    all_targets = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating {split_name}")
    for idx, (data_dict, target) in pbar: 
        data_dict['pos'] = data_dict['pos'].cuda(non_blocking=True)
        data_dict['x'] = data_dict['x'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits = model(data_dict)

        loss = criterion(logits.squeeze(-1), target.squeeze(-1))
        
        mse_meter.update(logits.squeeze(-1), target.squeeze(-1))
        mae_meter.update(logits.squeeze(-1), target.squeeze(-1))

        loss_meter.update(loss.item())
        
        # Store predictions and targets
        all_predictions.append(logits.squeeze(-1).cpu().detach().numpy())
        all_targets.append(target.squeeze(-1).cpu().detach().numpy())
    
    # Concatenate all collected arrays
    predictions_array = np.concatenate(all_predictions, axis=0)
    targets_array = np.concatenate(all_targets, axis=0)

    # Save to disk in the run's log directory
    pred_path = os.path.join(cfg.run_dir, f"{split_name}_predictions.npy")
    target_path = os.path.join(cfg.run_dir, f"{split_name}_targets.npy")
    
    np.save(pred_path, predictions_array)
    np.save(target_path, targets_array)
    logging.info(f"Saved {split_name} predictions to: {pred_path}")
    logging.info(f"Saved {split_name} targets to: {target_path}")

    final_mse = mse_meter.compute()
    final_mae = mae_meter.compute()
    final_loss = loss_meter.avg 

    return final_loss, final_mse, final_mae