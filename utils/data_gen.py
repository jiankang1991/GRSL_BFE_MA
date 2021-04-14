
import scipy.misc
import os
import numpy as np
from glob import glob

from PIL import Image


class Data_Gen:
    def __init__(self, img_dir, mask_dir, val_img_dir, val_mask_dir, transform=None, phase='train'):

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.val_img_dir = val_img_dir
        self.val_mask_dir = val_mask_dir

        self.phase = phase
        self.transform = transform

        if phase == 'train':
            self.img_pths = glob(os.path.join(img_dir, '*.png'))
        else:
            self.img_pths = glob(os.path.join(val_img_dir, '*.png'))

    def __getitem__(self, index):

        if self.phase == 'train':
            img_pth = self.img_pths[index]
            img_nm = os.path.basename(img_pth)
            mask_pth = os.path.join(self.mask_dir, img_nm)
        else:
            img_pth = self.img_pths[index]
            img_nm = os.path.basename(img_pth)
            mask_pth = os.path.join(self.val_mask_dir, img_nm)

        sat_img = np.array(Image.open(img_pth))
        mask_img = np.array(Image.open(mask_pth))

        mask = np.zeros((mask_img.shape[0], mask_img.shape[1]))
        mask[mask_img == 255] = 1

        mask = mask.astype(np.int32)

        sample = {'sat_img': sat_img, 'map_img': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_pths)









