import os, logging, csv, numpy as np # Removed wandb import
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
#from torch.utils.tensorboard import SummaryWriter # Keep this import if cfg.tensorboard is True, otherwise remove (based on final decision)

# OpenPoints utilities
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, load_checkpoint_inv, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, AverageMeter # Removed Wandb from here

# Dataset and Transforms
from torch.utils.data import DataLoader # Directly import DataLoader
from openpoints.transforms import build_transforms_from_cfg

# Optimizer, Scheduler, Model building
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.models import build_model_from_cfg

# Regression-specific metrics from torchmetrics
from torchmetrics import MeanSquaredError, MeanAbsoluteError 

# Your custom dataset and model components
from create_event_dataset import EventPointCloudDataset 
from openpoints.models.regression.reg_head import RegressionHead 
from RegressionModelWrapper import RegressionModelWrapper 


# --- Regression-specific helper functions ---
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
    
    # Logger setup (uses Python's built-in logging module for console & file)
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    
    # Initialize TensorBoard writer (based on cfg)
    if cfg.rank == 0 : 
        # REMOVED Wandb.launch call
        if cfg.common.tensorboard: # Only create SummaryWriter if tensorboard is True in config
            writer = SummaryWriter(log_dir=cfg.run_dir) 
        else:
            writer = None # Set writer to None if TensorBoard is disabled
    else:
        writer = None 

    set_random_seed(cfg.common.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    
    criterion = nn.MSELoss(reduction=cfg.loss.get('reduction', 'mean')).cuda() 

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
        use_transforms=cfg.dataset.common.get('transforms'), # Apply common transforms to the base dataset
        max_samples=cfg.dataset.common.get('max_samples') # Pass the max_samples from YAML here
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
    )
    val_loader = DataLoader(
        val_dataset_split,
        batch_size=cfg.dataset.val.get('batch_size', cfg.train.batch_size), 
        num_workers=cfg.dataset.val.get('num_workers', 4),
        shuffle=False, 
        drop_last=False 
    )
    test_loader = DataLoader(
        test_dataset_split,
        batch_size=cfg.get('test_batch_size', cfg.dataset.val.get('batch_size', cfg.train.batch_size)),
        num_workers=cfg.get('test_num_workers', 4),
        shuffle=False, 
        drop_last=False
    )

    logging.info(f"Length of training dataset: {len(train_loader.dataset)}")
    logging.info(f"Length of validation dataset: {len(val_loader.dataset)}")
    logging.info(f"Length of test dataset: {len(test_loader.dataset)}")
    
    cfg.num_outputs = 1 
    validate_fn = validate_regression 

    best_val_mse = float('inf') 
    best_epoch = 0

    model.zero_grad() 
    for epoch in range(cfg.train.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch) 
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1 
        
        train_loss, train_mse, train_mae = \
            train_one_epoch(model, train_loader, optimizer, scheduler, epoch, criterion, cfg)

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

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_loss {train_loss:.4f}, val_mse {val_mse:.4f}, best val mse {best_val_mse:.4f}')
        
        # --- TensorBoard Logging (Only if writer is not None) ---
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
        # --- End TensorBoard Logging ---

        if cfg.train.sched_on_epoch:
            scheduler.step(epoch) 
        if cfg.rank == 0: 
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val_mse': best_val_mse}, 
                            is_best=is_best
                            )
    
    # Test the last epoch model 
    test_loss_last, test_mse_last, test_mae_last = validate_regression(model, test_loader, criterion, cfg)
    print_regression_results(test_loss_last, test_mse_last, test_mae_last, cfg.epochs, cfg)
    if writer is not None:
        writer.add_scalar('test/loss_last_epoch', test_loss_last, cfg.epochs)
        writer.add_scalar('test/mse_last_epoch', test_mse_last, cfg.epochs)
        writer.add_scalar('test/mae_last_epoch', test_mae_last, cfg.epochs)

    # Test the best validataion model
    best_epoch_loaded, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_loss_best, test_mse_best, test_mae_best = validate_regression(model, test_loader, criterion, cfg)
    if writer is not None:
        writer.add_scalar('test/loss_best_val_epoch', test_loss_best, best_epoch_loaded)
        writer.add_scalar('test/mse_best_val_epoch', test_mse_best, best_epoch_loaded)
        writer.add_scalar('test/mae_best_val_epoch', test_mae_best, best_epoch_loaded)
    print_regression_results(test_loss_best, test_mse_best, test_mae_best, best_epoch_loaded, cfg)

    # Close TensorBoard writer (if active)
    if writer is not None:
        writer.close() 
    
    dist.destroy_process_group()


# --- Modified train_one_epoch function ---
def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, criterion, cfg): # Added criterion param
    loss_meter = AverageMeter()
    mse_meter = MeanSquaredError().to(cfg.rank)
    mae_meter = MeanAbsoluteError().to(cfg.rank)

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=len(train_loader)) 
    num_iter = 0
    for idx, (data_dict, target) in pbar: 
        data_dict['pos'] = data_dict['pos'].cuda(non_blocking=True)
        data_dict['x'] = data_dict['x'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True) 

        num_iter += 1
        
        logits = model(data_dict) 

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

        mse_meter.update(logits.squeeze(-1), target.squeeze(-1))
        mae_meter.update(logits.squeeze(-1), target.squeeze(-1))
        
        loss_meter.update(loss.item())
        if idx % cfg.train.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} MSE {mse_meter.compute():.4f} MAE {mae_meter.compute():.4f}")
    
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
        data_dict['pos'] = data_dict['pos'].cuda(non_blocking=True)
        data_dict['x'] = data_dict['x'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits = model(data_dict)

        loss = criterion(logits.squeeze(-1), target.squeeze(-1))
        
        mse_meter.update(logits.squeeze(-1), target.squeeze(-1))
        mae_meter.update(logits.squeeze(-1), target.squeeze(-1))

        loss_meter.update(loss.item())
        if idx % cfg.train.print_freq == 0:
            pbar.set_description(f"Val Epoch [{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} MSE {mse_meter.compute():.4f} MAE {mae_meter.compute():.4f}")
    
    val_mse = mse_meter.compute()
    val_mae = mae_meter.compute()
    val_loss = loss_meter.avg 

    return val_loss, val_mse, val_mae