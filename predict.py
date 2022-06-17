import argparse
import os
import pprint

import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim

from tomoseg import models, datasets
from tomoseg.config import update_config, config
from tomoseg.utils.utils import create_logger, get_model_summary, log_info
from torch.utils.data.distributed import DistributedSampler
from tomoseg.core import function
import time
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Predict with segmentation network')
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

    logger, final_output_dir, tb_log_dir = create_logger(config, 'predict')

    log_info(pprint.pformat(args))
    log_info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    device = torch.device(args.device)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpus)

    model = eval('models.' + config.MODEL.NAME)(
        input_nc=config.DATASET.NUM_INPUT_CHANNELS,
        output_nc=config.DATASET.NUM_CLASSES
    )

    dump_input = torch.rand(
        (1, config.DATASET.NUM_INPUT_CHANNELS, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    log_info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        state_dict = torch.load(config.TEST.MODEL_FILE)
        model_file = config.TEST.MODEL_FILE
    else:
        model_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        checkpoint = torch.load(model_file, map_location='cpu')
        state_dict = checkpoint['model']

    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}

    for k, _ in state_dict.items():
        log_info('Loading {} from pretrained model in: {}'.format(k, model_file))

    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

    # prepare data

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    unlabeled_dataset = eval('datasets.' + config.DATASET.DATASET)(
        split='unlabeled',
        list_path=config.DATASET.UNLABELED_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        num_input_channels=config.DATASET.NUM_INPUT_CHANNELS,
        preprocessing_fn=config.DATASET.PREPROCESSING_FN,
        multi_scale=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size)

    unlabeledloader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=1,  # ONLY SUPPORT BATCH SIZE 1
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    log_info(f'{len(unlabeled_dataset)} validation examples in {len(unlabeledloader)} batches')

    start_time = time.time()

    log_info('Predicting ...')
    function.predict(model, unlabeledloader, unlabeled_dataset, config, sv_dir=config.TEST.SV_DIR)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log_info(f'Prediction time {total_time_str}')


if __name__ == '__main__':
    main()
