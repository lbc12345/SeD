import os
import glob
import random
import numpy as np
import torch.utils.data as data
import cv2
from torchvision.transforms import ToTensor


class Train(data.Dataset):
    def __init__(self, scale=4, patch_size=64, data_root='./DF2K'):
        self.scale = scale
        self.patch_size = patch_size

        self.dir_hr = os.path.join(data_root, 'train_sub')
        self.dir_lr = os.path.join(data_root, 'train_sub_x4')
        self.images_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*.png'))
        )
        self.images_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*.png'))
        )

    def __getitem__(self, idx):

        filename = os.path.basename(self.images_hr[idx])
        hr = cv2.imread(self.images_hr[idx])  # BGR, n_channels=3
        lr = cv2.imread(self.images_lr[idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)  # RGB, n_channels=3
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        h, w, _ = lr.shape
        croph = np.random.randint(0, h - self.patch_size + 1)
        cropw = np.random.randint(0, w - self.patch_size + 1)
        
        hr = hr[croph*self.scale: croph*self.scale+self.patch_size*self.scale, cropw*self.scale: cropw*self.scale+self.patch_size*self.scale, :]
        lr = lr[croph: croph+self.patch_size, cropw: cropw+self.patch_size, :]

        mode = random.randint(0, 7)

        lr, hr = augment_img(lr, mode=mode), augment_img(hr, mode=mode)

        lr = ToTensor()(lr.copy())
        hr = ToTensor()(hr.copy())
        
        return lr, hr, filename

    def __len__(self):
        return len(self.images_hr)


class Test(data.Dataset):
    def __init__(self, data_root='./Evaluation'):

        self.dir_hr = os.path.join(data_root, 'Set5/GTmod12')
        self.dir_lr = os.path.join(data_root, 'Set5/LRbicx4')
        self.images_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*.png'))
        )
        self.images_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*.png'))
        )

    def __getitem__(self, idx):

        filename = os.path.basename(self.images_hr[idx])
        hr = cv2.imread(self.images_hr[idx])  # BGR, n_channels=3
        lr = cv2.imread(self.images_lr[idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)  # RGB, n_channels=3
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        lr = ToTensor()(lr.copy())
        hr = ToTensor()(hr.copy())

        return lr, hr, filename

    def __len__(self):
        return len(self.images_hr)


def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
