# src/data/dataset.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
Paired RGB-depth dataset for LDE.
Includes quadrant cropping, jitter, flips, and raw-depth min-max normalization to [0,1].
"""
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F

class _Gamma:
    """Applies a random gamma correction to an image."""
    def __call__(self, img):
        return F.adjust_gamma(img, random.uniform(0.7, 1.3))

class _Flicker:
    """Applies a random brightness flicker to an image."""
    def __call__(self, img):
        return F.adjust_brightness(img, random.uniform(0.9, 1.1))

class RGBDepthDataset(Dataset):
    """
    A dataset for loading paired RGB images and depth maps.

    Args:
        img_dir (str): Directory containing RGB images.
        dep_dir (str): Directory containing depth data (.npz files).
        size (int): The target size for output images and depth maps.
    """
    def __init__(self, img_dir, dep_dir, size=224):
        super().__init__()
        self.img_dir = img_dir
        self.dep_dir = dep_dir
        self.size = size

        # gather matching basenames
        img_files = {os.path.splitext(f)[0]
                     for f in os.listdir(img_dir)
                     if f.lower().endswith(('.jpg','.png','.jpeg'))}
        dep_files = {os.path.splitext(f)[0]
                     for f in os.listdir(dep_dir)
                     if f.lower().endswith('.npz')}
        self.names = sorted(img_files.intersection(dep_files))

        # normalization for RGB
        self.rgb_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])

        # color and blur augmentations
        self.jitter = T.Compose([
            T.ColorJitter(brightness=0.3,
                          contrast=0.3,
                          saturation=0.3,
                          hue=0.05),
            _Gamma(),
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.5),
            _Flicker()
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        rgb = Image.open(
            os.path.join(self.img_dir, name + '.jpg')
        ).convert('RGB')

        # load depth array
        try:
            arr = np.load(os.path.join(self.dep_dir, name + '.npz'))
            d = arr.get('inverse_depth',
                        arr.get('depth')).astype('float32')
        except Exception as e:
            raise IOError(f"Error loading depth file {name}.npz: {e}")

        depth = Image.fromarray(d)

        w, h = rgb.size

        # quadrant crop or resize
        op = random.randint(0,4)
        if op < 4 and w >= self.size and h >= self.size:
            left = 0 if op % 2 == 0 else w - self.size
            top = 0 if op < 2 else h - self.size
            rgb_c = rgb.crop((left, top,
                              left + self.size,
                              top + self.size))
            depth_c = depth.crop((left, top,
                                  left + self.size,
                                  top + self.size))
        else:
            rgb_c = rgb.resize((self.size, self.size),
                               Image.Resampling.BILINEAR)
            depth_c = depth.resize((self.size, self.size),
                                   Image.Resampling.NEAREST)

        # apply rgb jitter and normalize
        rgb_aug = self.jitter(rgb_c)
        rgb_t = self.rgb_norm(rgb_aug)

        # per-image depth min-max normalize
        da = np.array(depth_c, dtype='float32')
        dmin, dmax = da.min(), da.max()
        dn = (da - dmin) / (dmax - dmin + 1e-8)
        d_t = torch.from_numpy(dn).unsqueeze(0)

        # random flips
        if random.random() > 0.5:
            rgb_t = torch.fliplr(rgb_t)
            d_t = torch.fliplr(d_t)
        if random.random() > 0.5:
            rgb_t = torch.flipud(rgb_t)
            d_t = torch.flipud(d_t)

        return rgb_t.contiguous(), d_t.contiguous()
