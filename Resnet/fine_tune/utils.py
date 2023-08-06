import random
import torch

import numpy as np

import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# from munch import munchify

def worker_init_fn(worker_id):
    """
    Init worker in dataloader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def init_seed(args):
    """
    Set random seed for torch and numpy.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TransformImage(object):

    def __init__(self, opts):
        # if type(opts) == dict:
        #     opts = munchify(opts)
        # self.input_size = opts.input_size
        # self.input_space = opts.input_space
        # self.input_range = opts.input_range
        self.mean = opts['mean']
        self.std = opts['std']

        tfs = []

        # tfs.append(transforms.Resize((self.input_size[1], self.input_size[2])))
        #
        # tfs.append(transforms.ToTensor())
        # tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        # tfs.append(ToRange255(max(self.input_range)==255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor

class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, img):
        img = img.convert(self.space)
        return img

class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor
