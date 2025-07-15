# src/visual_test.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
Offline visual test - show RGB | predicted depth | ground truth.
"""
import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from network.model import LDE_Model

DEV     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIZE    = 224
CKPT    = 'checkpoints/lde_epoch68.pth'
IMG_DIR = 'dataset/images'
DEP_DIR = 'dataset/depth'

# only resize and normalize
tfm = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

def cmap_uint8(x):
    # map float depth [0,1] to color
    mn, mx = x.min(), x.max()
    y = ((x - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(y, cv2.COLORMAP_JET)

def load_gt(path):
    arr = np.load(path)
    d = arr.get('inverse_depth', arr.get('depth')).astype('float32')
    return d

def main():
    net = LDE_Model().to(DEV).eval()
    ck = torch.load(CKPT, map_location=DEV)
    net.load_state_dict(ck['net'])

    for img_path in sorted(glob.glob(os.path.join(IMG_DIR, '*.*'))):
        name = os.path.splitext(os.path.basename(img_path))[0]
        dp = os.path.join(DEP_DIR, name + '.npz')
        if not os.path.isfile(dp):
            continue

        rgb_pil = Image.open(img_path).convert('RGB')
        orig = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)
        h, w = orig.shape[:2]
        gt = load_gt(dp)

        inp = tfm(rgb_pil).unsqueeze(0).to(DEV)
        with torch.inference_mode():
            pred = net(inp)['pred'].squeeze().cpu().numpy()

        pred_full = cv2.resize(pred, (w,h), interpolation=cv2.INTER_NEAREST)
        gt_full   = cv2.resize(gt,   (w,h), interpolation=cv2.INTER_NEAREST)

        cp = cmap_uint8(pred_full)
        cg = cmap_uint8(gt_full)
        panel = np.hstack([orig, cp, cg])

        cv2.imshow('RGB | Predicted | Ground Truth', panel)
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
