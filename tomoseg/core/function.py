import os

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from tifffile import imsave


from tomoseg.utils import utils


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = utils.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, config, writer_dict, scaler=None):
    model.train()
    ave_loss = utils.AverageMeter()
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    for i_iter, batch in enumerate(data_loader):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        loss.mean()

        if utils.is_dist_avail_and_initialized():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        ave_loss.update(reduced_loss.item())
        if i_iter % config.PRINT_FREQ == 0:
            message = 'Epoch: {}/{} Iter: {}/{} LR: {:.6f} Batch Loss: {:.6f} Average Loss: {:.6f}'.format(
                epoch,
                config.TRAIN.EPOCHS,
                i_iter,
                len(data_loader),
                optimizer.param_groups[0]['lr'],
                reduced_loss.item(),
                ave_loss.average()
            )
            utils.log_info(message)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
    return ave_loss.average()


def validate(model, criterion, data_loader, device, config, metrics, writer_dict):
    model.eval()
    ave_loss = utils.AverageMeter()
    scores = {}
    if 'miou' in metrics:
        confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.inference_mode():
        for i_iter, batch in enumerate(tqdm(data_loader)):
            images, labels, _, _ = batch
            size = labels.size()
            images = images.to(device)
            labels = labels.long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.mean()

            if utils.is_dist_avail_and_initialized():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss

            ave_loss.update(reduced_loss.item())
            if 'miou' in metrics:
                confusion_matrix += utils.get_confusion_matrix(
                    labels,
                    outputs,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

    if utils.is_dist_avail_and_initialized():
        if 'miou' in metrics:
            confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
            reduced_confusion_matrix = reduce_tensor(confusion_matrix)
            confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    if 'miou' in metrics:
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        scores['miou'] = [round(mean_IoU, 4)] + [round(iou, 4) for iou in IoU_array]

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    if 'miou' in metrics:
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), scores


def test(model, data_loader, dataset, config, metrics, sv_dir=None):
    model.eval()
    scores = {}
    if 'miou' in metrics:
        confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.inference_mode():
        for i_iter, batch in enumerate(tqdm(data_loader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = dataset.multi_scale_inference(
                model,
                image,
                scales=config.TEST.SCALE_LIST)

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=True
                )
            if 'miou' in metrics:
                confusion_matrix += utils.get_confusion_matrix(
                    label,
                    pred,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL)
            if sv_dir:
                pred = pred.detach().cpu().numpy()[0]
                pred = np.argmax(pred, 0)
                pred = pred.astype('uint8')
                sv_path = os.path.join(sv_dir, '{}_predictions'.format(config.EXPERIMENT_NAME))
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                imsave(os.path.join(sv_path, name[0]+'.png'), pred)

    if 'miou' in metrics:
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        scores['miou'] = [round(mean_IoU, 4)] + [round(iou, 4) for iou in IoU_array]

    return scores


def predict(model, data_loader, dataset, config, sv_dir=None):
    model.eval()
    with torch.inference_mode():
        for i_iter, batch in enumerate(tqdm(data_loader)):
            image, size, name = batch
            size = size[0]
            pred = dataset.multi_scale_inference(
                model,
                image,
                scales=config.TEST.SCALE_LIST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, tuple(size[:2].numpy()),
                    mode='bilinear', align_corners=True
                )

            if sv_dir:
                pred = pred.detach().cpu().numpy()[0]
                pred = np.argmax(pred, 0)
                pred = pred.astype('uint8')
                sv_path = os.path.join(sv_dir, '{}_predictions'.format(config.EXPERIMENT_NAME))
                if not os.path.exists(sv_path):
                    os.makedirs(sv_path, exist_ok=True)
                imsave(os.path.join(sv_path, name[0]+'.png'), pred)
