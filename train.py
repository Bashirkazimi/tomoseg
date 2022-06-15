# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import math
import os
import pprint

import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

from tomoseg import models, datasets
from tomoseg.config import update_config, config
from tomoseg.utils.utils import create_logger, get_model_summary, init_distributed_mode, log_info
from torch.utils.data.distributed import DistributedSampler
from tomoseg.core.losses import OhemCrossEntropy, CrossEntropy
from tomoseg.core import function
import time
import datetime
from tomoseg.utils import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    if args.seed > 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(config, 'train')

    log_info(pprint.pformat(args))
    log_info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    device = torch.device(args.device)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpus)
    distributed = init_distributed_mode()

    model = eval('models.' + config.MODEL.NAME)(
        input_nc=config.DATASET.NUM_INPUT_CHANNELS,
        output_nc=config.DATASET.NUM_CLASSES,
        pretrained=config.MODEL.PRETRAINED,
        init_fn=config.TRAIN.INIT_FN
    )

    dump_input = torch.rand(
        (1, config.DATASET.NUM_INPUT_CHANNELS, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    log_info(get_model_summary(model.cuda(), dump_input.cuda()))

    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        split='train',
        list_path=config.DATASET.TRAIN_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        num_input_channels=config.DATASET.NUM_INPUT_CHANNELS,
        preprocessing_fn=config.DATASET.PREPROCESSING_FN,
        multi_scale=config.TRAIN.MULTI_SCALE,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    log_info(f'{len(train_dataset)} training examples in {len(trainloader)} batches')

    val_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    val_dataset = eval('datasets.' + config.DATASET.DATASET)(
        split='val',
        list_path=config.DATASET.VAL_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        num_input_channels=config.DATASET.NUM_INPUT_CHANNELS,
        preprocessing_fn=config.DATASET.PREPROCESSING_FN,
        multi_scale=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=val_size)

    val_sampler = DistributedSampler(val_dataset) if distributed else None
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=val_sampler)

    log_info(f'{len(val_dataset)} validation examples in {len(valloader)} batches')

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights)

    model.to(device)
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        model_without_ddp = model.module
    else:
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    params_to_optimize = [{'params': params}]

    # optimizer
    if config.TRAIN.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params_to_optimize,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    elif config.TRAIN.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(params_to_optimize,
                                    lr=config.TRAIN.LR,
                                    weight_decay=config.TRAIN.WD
                                    )
    elif config.TRAIN.OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(params_to_optimize,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD
                                    )
    else:
        raise ValueError(f'Only support SGD, Adam, and RMSprop, not {config.TRAIN.OPTIMIZER}')

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(trainloader)
    main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (iters_per_epoch * (config.TRAIN.EPOCHS - config.TRAIN.LR_WARMUP_EPOCHS))) ** 0.9
    )

    if config.TRAIN.LR_WARMUP_EPOCHS > 0:
        warmup_iters = iters_per_epoch * config.TRAIN.LR_WARMUP_EPOCHS
        if config.TRAIN.LR_WARMUP_METHOD.lower() == 'linear':
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=config.TRAIN.LR_WARMUP_DECAY, total_iters=warmup_iters
            )
        elif config.TRAIN.LR_WARMUP_METHOD.lower() == 'constant':
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=config.TRAIN.LR_WARMUP_DECAY, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f'Invalid warmup lr method "{config.TRAIN.LR_WARMUP_METHOD.lower()}". Only linear and constant are supported.'
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    best_scores = {
        metric: {'value': (-math.inf if hib else math.inf), 'higher_is_better': hib}
        for metric, hib in zip(config.METRIC.METRICS, config.METRIC.HIGHER_IS_BETTER)
    }

    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location='cpu')
            for metric in best_scores.keys():
                best_scores[metric]['value'] = checkpoint['scores'][metric]
            last_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            log_info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start_time = time.time()
    for epoch in range(last_epoch, config.TRAIN.EPOCHS):
        if distributed:
            train_sampler.set_epoch(epoch)
        train_loss = function.train_one_epoch(model, criterion, optimizer, trainloader, lr_scheduler, device, epoch,
                                              config, writer_dict, scaler)
        current_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        message = 'Epoch: {}/{} Train Loss: {:.6f} Current Time: {}'.format(
            epoch,
            config.TRAIN.EPOCHS,
            train_loss,
            current_time
        )
        log_info(message)
        log_info('Validating ...')
        valid_loss, scores = function.validate(model, criterion, valloader, device, config, config.METRIC.METRICS,
                                               writer_dict)
        current_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        message = 'Epoch: {}/{} Valid Loss: {:.6f} Current Time: {} '.format(
            epoch,
            config.TRAIN.EPOCHS,
            valid_loss,
            current_time
        )

        for metric in scores.keys():
            if best_scores[metric]['higher_is_better'] and best_scores[metric]['value'] < scores[metric][0]:
                best_scores[metric]['value'] = scores[metric][0]
                file_path = os.path.join(final_output_dir, 'best_{}.pth'.format(metric))
                utils.save_on_master(model_without_ddp.state_dict(), file_path)
            if not best_scores[metric]['higher_is_better'] and best_scores[metric]['value'] > scores[metric][0]:
                best_scores[metric]['value'] = scores[metric][0]
                file_path = os.path.join(final_output_dir, 'best_{}.pth'.format(metric))
                utils.save_on_master(model_without_ddp.state_dict(), file_path)
            current_metric_message = 'Current {}: {}, Best {}: {}, Per Class {}:{} '.format(
                metric,
                scores[metric][0],
                metric,
                best_scores[metric]['value'],
                metric,
                scores[metric][1:]
            )
            message = message + current_metric_message
        log_info(message)
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scores': {metric: score[0] for metric, score in scores.items()}
        }
        if args.amp:
            checkpoint['scaler'] = scaler.state_dict()
        utils.save_on_master(checkpoint, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log_info(f'Training time {total_time_str}')


if __name__ == '__main__':
    main()
