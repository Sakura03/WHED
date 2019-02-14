from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2

def prepare_image(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def prepare_image_cv2(im):
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

class BSDSLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)

            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb < 64] = 0.0
            lb[lb >= 64] = 1.0
            
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
                
            if len(lb.nonzero()[0]) > 10:
                h, w = lb.shape[1], lb.shape[2]
                test = True
                while test:
                    idx_i = np.random.choice(h-99)
                    idx_j = np.random.choice(w-99)
                    lb_crop = lb[:, idx_i:idx_i+100, idx_j:idx_j+100]
                    test = np.all(lb_crop == 0)
                    img_crop = img[:, idx_i:idx_i+100, idx_j:idx_j+100]
                return img_crop, lb_crop
            else:
                return img, lb
        else:
            img_file = self.filelist[index].rstrip()
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img

