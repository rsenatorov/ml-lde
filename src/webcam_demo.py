# src/webcam_demo.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
Live webcam demo - center crop then infer and display depth.
"""
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from network.model import LDE_Model

DEV       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IN_SIZE   = 224
CROP_SIZE = 448
CKPT      = 'checkpoints/lde_epoch68.pth'

# resize and normalize for model
tfm = transforms.Compose([
    transforms.Resize((IN_SIZE, IN_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

def cmap_uint8(x):
    mn, mx = x.min(), x.max()
    y = ((x - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(y, cv2.COLORMAP_JET)

def main():
    net = LDE_Model().to(DEV).eval()
    ck = torch.load(CKPT, map_location=DEV)
    net.load_state_dict(ck['net'])

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError('Webcam failed to open')

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        top  = max(0, (h - CROP_SIZE)//2)
        left = max(0, (w - CROP_SIZE)//2)
        crop = frame[top:top+CROP_SIZE, left:left+CROP_SIZE]

        rgb_pil = Image.fromarray(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        )
        inp = tfm(rgb_pil).unsqueeze(0).to(DEV)

        with torch.inference_mode():
            pred = net(inp)['pred'].squeeze().cpu().numpy()

        pred_up = cv2.resize(pred,
                             (CROP_SIZE, CROP_SIZE),
                             interpolation=cv2.INTER_NEAREST)
        cp = cmap_uint8(pred_up)
        panel = np.hstack([crop, cp])

        cv2.imshow('Webcam | Depth', panel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
