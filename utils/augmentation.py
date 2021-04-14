
import numpy as np
from torchvision import transforms
import random
import torch

from PIL import ImageEnhance
from PIL import Image

from scipy.ndimage import distance_transform_edt as distance
from torch import Tensor, einsum
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union


class RandomCropTarget(object):
    """
    Crop the image and target randomly in a sample.

    Args:
    output_size (tuple or int): Desired output size. If int, square crop
        is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        sat_img, map_img = sample['sat_img'], sample['map_img']

        h, w = sat_img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sat_img = sat_img[top: top + new_h, left: left + new_w]
        map_img = map_img[top: top + new_h, left: left + new_w]

        return {'sat_img': sat_img, 'map_img': map_img}

class CenterCropTarget(object):
    """
    Crop the image and target in the center in a sample.

    Args:
    output_size (tuple or int): Desired output size. If int, square crop
        is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        sat_img, map_img = sample['sat_img'], sample['map_img']

        h, w = sat_img.shape[:2]
        new_h, new_w = self.output_size

        i = int(round((h - new_h) / 2.))
        j = int(round((w - new_w) / 2.))


        sat_img = sat_img[i: i + new_h, j: j + new_w]
        map_img = map_img[i: i + new_h, j: j + new_w]

        return {'sat_img': sat_img, 'map_img': map_img}

class RandomRotate(object):

    def __call__(self, sample):
        
        rand = random.random()

        if rand < 0.25:
            sat_img = np.rot90(sample['sat_img'], k=1)
            map_img = np.rot90(sample['map_img'], k=1)

        elif 0.25 <= rand and rand < 0.5:
            sat_img = np.rot90(sample['sat_img'], k=2)
            map_img = np.rot90(sample['map_img'], k=2)

        elif 0.5 <= rand and rand < 0.75:
            sat_img = np.rot90(sample['sat_img'], k=3)
            map_img = np.rot90(sample['map_img'], k=3)

        elif 0.75 <= rand and rand < 1:
            sat_img = sample['sat_img']
            map_img = sample['map_img']
            
        return {'sat_img': sat_img.copy(), 'map_img': map_img.copy()}

class RandomFlip(object):

    def __call__(self, sample):
        
        rand = random.random()

        if rand < 1 / 3.0:
            sat_img = np.fliplr(sample['sat_img'])
            map_img = np.fliplr(sample['map_img'])

        elif 1 / 3.0 <= rand and rand < 2 / 3.0:
            sat_img = np.flipud(sample['sat_img'])
            map_img = np.flipud(sample['map_img'])

        elif 2 / 3.0 <= rand and rand < 1:
            sat_img = sample['sat_img']
            map_img = sample['map_img']
            
        return {'sat_img': sat_img.copy(), 'map_img': map_img.copy()}

class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        # print(type(sat_img))
        return {'sat_img': transforms.functional.to_tensor(sat_img),
                'map_img': torch.from_numpy(map_img).long()} # unsqueeze for the channel dimension

class Normalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']

        for t, m, s in zip(sat_img, self.mean, self.std):
            t.sub_(m).div_(s)

        return {'sat_img': sat_img,
                'map_img': map_img}


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())
    
def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    # assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def class2one_hot(seg: np.array, C: int) -> np.array:
    # if len(seg.shape) == 2:  # Only w, h, used by the dataloader
    #     seg = np.expand_dims(seg, axis=0)
    # assert sset(seg, list(range(C)))

    # b, w, h = seg.shape  # type: Tuple[int, int, int]
    res = np.stack([seg == c for c in range(C)], axis=0).astype(np.int32)

    # print(res.shape)
    return res

# def class2one_hot(seg: Tensor, C: int) -> Tensor:
#     if len(seg.shape) == 2:  # Only w, h, used by the dataloader
#         seg = seg.unsqueeze(dim=0)
#     assert sset(seg, list(range(C)))

#     b, w, h = seg.shape  # type: Tuple[int, int, int]

#     res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
#     assert res.shape == (b, C, w, h)
#     assert one_hot(res)

#     return res


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sat_img):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        # print(type(sat_img))
        return transforms.functional.to_tensor(sat_img)


class ToTensorTargetDist(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, cls_num):
        self.cls_num = cls_num
    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']
        onehot_img = class2one_hot(map_img, self.cls_num)
        dist_img = one_hot2dist(onehot_img)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        # print(type(sat_img))
        return {'sat_img': transforms.functional.to_tensor(sat_img),
                'map_img': torch.from_numpy(map_img).long(),
                'dist_img':torch.from_numpy(dist_img).type(torch.float32)} # unsqueeze for the channel dimension



























