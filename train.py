import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist


#from torch.utils.tensorboard import SummaryWriter
from PointNeXt.openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, load_checkpoint_inv, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
#from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
#from PointNeXt.openpoints.dataset import build_dataloader_from_cfg
from PointNeXt.openpoints.transforms import build_transforms_from_cfg
from PointNeXt.openpoints.optim import build_optimizer_from_cfg
from PointNeXt.openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from PointNeXt.openpoints.models import build_model_from_cfg
from PointNeXt.openpoints.models.layers import furthest_point_sample, fps

from torchmetrics import MeanSquaredError, MeanAbsoluteError 
from EventPointCloudDataset import EventPointCloudDataset
from openpoints.models.regression.regression_head import RegressionHead
from openpoints.models.regression_model_wrapper import RegressionModelWrapper

#work pls
"""
def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()
"""

def print_regression_results(loss, mse, mae, epoch, cfg):
	s = f'\nE@{epoch}\tLoss: {loss:.4f}\tMSE: {mse:.4f}\tMAE: {mae:.4f}\n'
	logging.info(s)
"""
def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()
"""
"""
def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)
"""

def main(gpu, cfg, profile=False):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # 
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
	    
        writer = None # SummaryWriter(log_dir=cfg.run_dir)
	    
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
	
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

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.get('num_workers', 4), 
        shuffle=True, # Shuffle for training
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val.get('batch_size', cfg.train.batch_size), 
        num_workers=cfg.val.get('num_workers', 4),
        shuffle=False, # No shuffle for validation
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get('test_batch_size', cfg.val.get('batch_size', cfg.train.batch_size)),
        num_workers=cfg.get('test_num_workers', 4),
        shuffle=False, # No shuffle for testing
        drop_last=False
    )

    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    logging.info(f"length of test dataset: {len(test_loader.dataset)}")
    
    # Removed: num_classes, num_points assertions (no num_classes for regression)
    # Removed: cfg.classes = ... (no classes for regression)
    
    # Set the number of outputs for the model, which is 1 for opang regression
    cfg.num_outputs = 1 
    
    # Ensure validate_fn points to our new regression validation function
    validate_fn = validate_regression

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder_inv':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint_inv(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    # ===> start training
    best_val_mse = float('inf') 
    best_epoch = 0
    model.zero_grad()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        train_loss, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader,
                            optimizer, scheduler, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_macc, val_oa, val_accs, val_cm = validate_fn(
                model, val_loader, cfg)
            is_best = val_oa > best_val
            if is_best:
                best_val = val_oa
                macc_when_best = val_macc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
                print_regression_results(val_oa, val_macc, val_accs, epoch, cfg)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_oa {train_oa:.2f}, val_oa {val_oa:.2f}, best val oa {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_mse', train_mse, epoch)
            writer.add_scalar('train_mae', train_mae, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_mse', val_mse, epoch)
            writer.add_scalar('val_mae', val_mae, epoch)
            writer.add_scalar('best_val_mse', best_val_mse, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
    # test the last epoch
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    print_regression_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if writer is not None:
        writer.add_scalar('test_loss_last_epoch', test_loss_last, cfg.epochs)
        writer.add_scalar('test_mse_last_epoch', test_mse_last, cfg.epochs)
        writer.add_scalar('test_mae_last_epoch', test_mae_last, cfg.epochs)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_loss_best_val_epoch', test_loss_best, best_epoch_loaded)
        writer.add_scalar('test_mse_best_val_epoch', test_mse_best, best_epoch_loaded)
        writer.add_scalar('test_mae_best_val_epoch', test_mae_best, best_epoch_loaded)
    print_regression_results(test_oa, test_macc, test_accs, best_epoch, cfg)

    if writer is not None:
        writer.close()
    dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    npoints = cfg.num_points

    mse_meter = MeanSquaredError().to(cfg.rank)
    mae_meter = MeanAbsoluteError().to(cfg.rank)

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=len(train_loader)) # Use len() instead of __len__()
    num_iter = 0
    for idx, (data_dict, target) in pbar: # Unpack data_dict and target
        # Move data to GPU
        data_dict['pos'] = data_dict['pos'].cuda(non_blocking=True)
        data_dict['x'] = data_dict['x'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True) # Target is already float

        num_iter += 1
        
        # --- REMOVED / COMMENTED OUT: Original point resampling block ---
        # Your EventPointCloudDataset is now responsible for sampling/padding to cfg.num_points.
        # This block often causes issues if the input data format doesn't exactly match
        # its assumptions (e.g., points[:,:,:3]).
        # num_curr_pts = points.shape[1] 
        # if num_curr_pts > npoints: 
        #    ... (rest of resampling) ...
        # data['pos'] = points[:, :, :3].contiguous()
        # data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

        # Forward pass: model expects data_dict (or pos, x directly if wrapper wasn't used)
        # Our RegressionModelWrapper's forward method expects data_dict
        logits = model(data_dict) # logits is now the regression output

        # Calculate loss (logits is output, target is ground truth)
        loss = criterion(logits.squeeze(-1), target.squeeze(-1)) # Ensure both (batch_size,) for MSELoss

        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # Update regression metrics
        mse_meter.update(logits.squeeze(-1), target.squeeze(-1))
        mae_meter.update(logits.squeeze(-1), target.squeeze(-1))
        
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} MSE {mse_meter.compute():.4f} MAE {mae_meter.compute():.4f}")
    
    # Final metrics for the epoch
    train_mse = mse_meter.compute()
    train_mae = mae_meter.compute()
    
    return loss_meter.avg, train_mse, train_mae

@torch.no_grad()
def validate_regression(model, val_loader, criterion, cfg): # Added criterion param
    model.eval() # set model to eval mode
    loss_meter = AverageMeter()
    mse_meter = MeanSquaredError().to(cfg.rank)
    mae_meter = MeanAbsoluteError().to(cfg.rank)

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for idx, (data_dict, target) in pbar: # Unpack data_dict and target
        # Move data to GPU
        data_dict['pos'] = data_dict['pos'].cuda(non_blocking=True)
        data_dict['x'] = data_dict['x'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Removed: Original resampling block (as dataset handles num_points)
        # points = data['x']
        # points = points[:, :npoints]
        # data['pos'] = points[:, :, :3].contiguous()
        # data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

        # Forward pass
        logits = model(data_dict) # logits is regression output

        # Calculate loss
        loss = criterion(logits.squeeze(-1), target.squeeze(-1))
        
        # Update metrics
        mse_meter.update(logits.squeeze(-1), target.squeeze(-1))
        mae_meter.update(logits.squeeze(-1), target.squeeze(-1))

        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Val Epoch [{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} MSE {mse_meter.compute():.4f} MAE {mae_meter.compute():.4f}")
    
    # Final metrics for validation
    val_mse = mse_meter.compute()
    val_mae = mae_meter.compute()
    val_loss = loss_meter.avg # Get average loss for the validation set

    return val_loss, val_mse, val_mae # Return loss, mse, mae

@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        points = data['x']
        points = points[:, :npoints]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    tp, count = cm.tp, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return macc, overallacc, accs, cm
