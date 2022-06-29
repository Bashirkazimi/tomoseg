from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'output'
_C.EXPERIMENT_NAME = 'default'
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet'
_C.MODEL.PRETRAINED = ''

_C.LOSS = CN()
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [1]

_C.METRIC = CN()
_C.METRIC.METRICS = ['miou']
_C.METRIC.HIGHER_IS_BETTER = [True]


# DATASET related params
_C.DATASET = CN()
_C.DATASET.DATASET = 'tomo'
_C.DATASET.NUM_INPUT_CHANNELS = 1
_C.DATASET.NUM_CLASSES = 4
_C.DATASET.TRAIN_SET = '/path/to/train.txt'  # file with list of input/output pairs, e.g. image1.png mask1.png
_C.DATASET.VAL_SET = '/path/to/val.txt'  # file with list of input/output pairs, e.g. image1.png mask1.png
_C.DATASET.TEST_SET = '/path/to/test.txt'  # file with list of input/output pairs, e.g. image1.png mask1.png
_C.DATASET.UNLABELED_SET = '/path/to/unlabeled.txt'  # file with list of unlabeled input images, e.g., image1.png
_C.DATASET.CLASS_WEIGHTS = None
_C.DATASET.PREPROCESSING_FN = None

# training
_C.TRAIN = CN()


_C.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.RANDOM_BRIGHTNESS = False
_C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001
_C.TRAIN.INIT_FN = None

_C.TRAIN.OPTIMIZER = 'SGD'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.LR_WARMUP_EPOCHS = 0
_C.TRAIN.LR_WARMUP_METHOD = 'linear'
_C.TRAIN.LR_WARMUP_DECAY = 0.01
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.EPOCHS = 484
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
# only using some training samples
_C.TRAIN.NUM_SAMPLES = 0

# testing
_C.TEST = CN()

_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048

_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
_C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False
_C.TEST.SCALE_LIST = [1]
_C.TEST.SV_DIR = None

_C.TEST.SAMPLE_IMAGES_PATH = None
_C.TEST.SAMPLE_LABELS_PATH = None
_C.TEST.SAMPLE_PREPROCESS = True
_C.TEST.SAMPLE_EXT = '.tif'



def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

