import os
import glob
import torch.utils.data as data
import cv2
from torchvision.transforms import ToTensor


class Test(data.Dataset):
    def __init__(self, data_lr_root='./', use_hr=True, data_hr_root=None):
        self.use_hr = use_hr
        if use_hr:
            assert data_hr_root != None, 'Please input your hr root!'
            self.dir_hr = data_hr_root
            self.images_hr = sorted(
                glob.glob(os.path.join(self.dir_hr, '*.png'))
            )
        self.dir_lr = data_lr_root
        self.images_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*.png'))
        )
        assert len(self.images_lr) != 0, 'There are no images in your lr root! Or your lr images are not end with .png'

    def __getitem__(self, idx):

        filename = os.path.basename(self.images_lr[idx])
        if self.use_hr:
            hr = cv2.imread(self.images_hr[idx])  # BGR, n_channels=3
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)  # RGB, n_channels=3
            hr = ToTensor()(hr.copy())
        
        lr = cv2.imread(self.images_lr[idx])
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        lr = ToTensor()(lr.copy())

        if self.use_hr:
            return {'lr': lr, 'hr': hr, 'fn': filename}
        return {'lr': lr, 'fn': filename}

    def __len__(self):
        return len(self.images_lr)
