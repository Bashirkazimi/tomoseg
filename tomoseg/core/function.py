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


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, config, writer_dict):
    model.train()
    ave_loss = utils.AverageMeter()
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    for i_iter, batch in enumerate(data_loader):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)
        losses, _ = model(images, labels)
        loss = losses.mean()

        if utils.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

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


def validate(model, data_loader, device, config, metrics, writer_dict):
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

            losses, preds = model(images, labels)

            loss = losses.mean()

            if utils.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss

            ave_loss.update(reduced_loss.item())

            if 'miou' in metrics:
                confusion_matrix += utils.get_confusion_matrix(
                    labels,
                    preds,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

    if utils.is_distributed():
        if 'miou' in metrics:
            confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
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


def evaluate_multiview(model, dataset, config, images, labels, first_axes=None, second_axes=None, return_sample=False, sv_dir=None):
    n, height, width = images.shape
    predicted3D = np.zeros((n, dataset.num_classes, height, width))

    if first_axes is None:
        first_axes = [
            (1, 0, 2),
            (2, 0, 1),
        ]
    if second_axes is None:
        second_axes = [
            (2, 1, 0, 3),
            (2, 1, 3, 0)
        ]

    model.eval()

    with torch.inference_mode():
        for i_iter in tqdm(range(n)):
            image = images[i_iter]
            image = np.expand_dims(image, (0, 1))
            # image = torch.from_numpy(image).float()
            image = torch.Tensor(image)
            pred = dataset.multi_scale_inference(
                model,
                image,
                scales=config.TEST.SCALE_LIST)

            if pred.size()[-2] != height or pred.size()[-1] != width:
                pred = F.interpolate(
                    pred, (height, width),
                    mode='bilinear', align_corners=True
                )

            pred = pred.detach().cpu().numpy()[0]
            predicted3D[i_iter] = pred

        for (ax1, ax2) in zip(first_axes, second_axes):
            current_images = np.transpose(images, ax1)
            n, height, width = current_images.shape
            current_predicted3D = np.zeros((n, dataset.num_classes, height, width))
            for i_iter in tqdm(range(n)):
                image = current_images[i_iter]
                image = np.expand_dims(image, (0, 1))
                image = torch.Tensor(image)

                pred = dataset.multi_scale_inference(
                    model,
                    image,
                    scales=config.TEST.SCALE_LIST)

                if pred.size()[-2] != height or pred.size()[-1] != width:
                    pred = F.interpolate(
                        pred, (height, width),
                        mode='bilinear', align_corners=True
                    )

                pred = pred.detach().cpu().numpy()[0]
                current_predicted3D[i_iter] = pred

            current_predicted3D = np.transpose(current_predicted3D, ax2)
            predicted3D += current_predicted3D

    predicted3D /= (len(first_axes) + 1)

    confusion_matrix = np.zeros((dataset.num_classes, dataset.num_classes))
    for i in tqdm(range(predicted3D.shape[0])):
        pred = np.expand_dims(predicted3D[i], 0)
        lbl = np.expand_dims(labels[i], 0)
        confusion_matrix += utils.get_confusion_matrix(
            torch.from_numpy(lbl).cuda(),
            torch.from_numpy(pred).cuda(),
            torch.from_numpy(pred).size(),
            dataset.num_classes,
            config.TRAIN.IGNORE_LABEL
        )

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    scores = [round(mean_IoU, 4)] + [round(iou, 4) for iou in IoU_array]

    predicted3D = np.argmax(predicted3D, 1)

    if sv_dir:
        os.makedirs(sv_dir, exist_ok=True)
        for i_iter in tqdm(range(predicted3D.shape[0])):
            prediction = predicted3D[i_iter].astype('float32')
            file_path = os.path.join(sv_dir, '{}.tif'.format(str(i_iter).zfill(5)))
            imsave(file_path, prediction)

    if return_sample:
        return scores, predicted3D

    return scores


def predict_multiview(model, dataset, config, images, first_axes=None, second_axes=None, return_sample=False, sv_dir=None):
    n, height, width = images.shape
    predicted3D = np.zeros((n, dataset.num_classes, height, width))

    if first_axes is None:
        first_axes = [
            (1, 0, 2),
            (2, 0, 1)
        ]
    if second_axes is None:
        second_axes = [
            (2, 1, 0, 3),
            (2, 1, 3, 0)
        ]

    model.eval()

    with torch.inference_mode():
        for i_iter in tqdm(range(n)):
            image = images[i_iter]
            image = np.expand_dims(image, (0, 1))
            # image = torch.from_numpy(image).float()
            image = torch.Tensor(image)
            pred = dataset.multi_scale_inference(
                model,
                image,
                scales=config.TEST.SCALE_LIST)

            if pred.size()[-2] != height or pred.size()[-1] != width:
                pred = F.interpolate(
                    pred, (height, width),
                    mode='bilinear', align_corners=True
                )

            pred = pred.detach().cpu().numpy()[0]
            predicted3D[i_iter] = pred

        for (ax1, ax2) in zip(first_axes, second_axes):
            current_images = np.transpose(images, ax1)
            n, height, width = current_images.shape
            current_predicted3D = np.zeros((n, dataset.num_classes, height, width))
            for i_iter in tqdm(range(n)):
                image = current_images[i_iter]
                image = np.expand_dims(image, (0, 1))
                image = torch.Tensor(image)

                pred = dataset.multi_scale_inference(
                    model,
                    image,
                    scales=config.TEST.SCALE_LIST)

                if pred.size()[-2] != height or pred.size()[-1] != width:
                    pred = F.interpolate(
                        pred, (height, width),
                        mode='bilinear', align_corners=True
                    )

                pred = pred.detach().cpu().numpy()[0]
                current_predicted3D[i_iter] = pred

            current_predicted3D = np.transpose(current_predicted3D, ax2)
            predicted3D += current_predicted3D

    predicted3D /= (len(first_axes) + 1)
    predicted3D = np.argmax(predicted3D, 1)

    if sv_dir:
        os.makedirs(sv_dir, exist_ok=True)
        for i_iter in tqdm(range(predicted3D.shape[0])):
            prediction = predicted3D[i_iter].astype('float32')
            file_path = os.path.join(sv_dir, '{}.tif'.format(str(i_iter).zfill(5)))
            imsave(file_path, prediction)

    if return_sample:
        return predicted3D
