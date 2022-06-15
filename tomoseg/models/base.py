import torch.nn as nn
import logging
from torch.nn import BatchNorm2d
import os
import torch
logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def init_weights(self, pretrained='', init_fn=None):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            update_dict = dict()
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and model_dict[k].shape == pretrained_dict[k].shape:
                        update_dict[k] = v
            for k, _ in update_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(update_dict)
            self.load_state_dict(model_dict)
        else:
            if init_fn is None:
                logger.info('=> init weights from normal distribution')
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, std=0.001)
                    elif isinstance(m, BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            else:
                # logger.info('=> init weights with {}'.format(init_fn))
                print('=> init weights with {}'.format(init_fn))
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        eval('nn.init.'+init_fn)(m.weight)
                    elif isinstance(m, BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

