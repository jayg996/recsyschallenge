import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
from utils import logger

def adjusting_learning_rate(optimizer, factor=.5, min_lr=0.00001):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = max(old_lr*factor, min_lr)
        param_group['lr'] = new_lr
        logger.info('adjusting learning rate from %.6f to %.6f' % (old_lr, new_lr))