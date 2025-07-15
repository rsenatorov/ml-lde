# src/train.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
Main training script for LDE (Light Depth Estimation) model.
"""
import os
import time
import math
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from data.dataset import RGBDepthDataset
from network.model import LDE_Model
from network.losses import LDELoss
from checkpoint import save_checkpoint, load_latest_checkpoint

# --- CONFIGURATION ---
CONFIG = {
    'image_dir':      'dataset/images',
    'depth_dir':      'dataset/depth',
    'val_split':      0.1,
    'batch_size':     8,
    'num_workers':    4,
    'lr':             1e-4,
    'weight_decay':   1e-2,
    'mixed_precision': True,
    'accum_steps':    2,
    'grad_clip':      1.0,
    'warmup_steps':   1000,
    'T_max':          2500000,
    'output_dir':     'checkpoints',
    'log_dir':        'logs',
    'quick_test':     False,
    'quick_fraction': 0.01
}

class CosineWarmup(optim.lr_scheduler._LRScheduler):
    """Cosine decay with linear warmup."""
    def __init__(self, optimizer,
                 warmup_steps, t_max,
                 eta_min=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_max = t_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps
                    for base_lr in self.base_lrs]
        p = (step - self.warmup_steps) / (self.t_max - self.warmup_steps)
        return [self.eta_min +
                (base_lr - self.eta_min) *
                0.5 * (1 + math.cos(math.pi * p))
                for base_lr in self.base_lrs]

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)

    # --- DATASET ---
    full_ds = RGBDepthDataset(CONFIG['image_dir'],
                              CONFIG['depth_dir'])
    if CONFIG['quick_test']:
        n = int(len(full_ds) * CONFIG['quick_fraction'])
        dataset = Subset(full_ds, range(n))
        print(f"[INFO] Quick test: {n} samples.")
    else:
        dataset = full_ds

    val_size = int(CONFIG['val_split'] * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train {train_size}, Val {val_size}")

    tl = DataLoader(train_ds,
                    batch_size=CONFIG['batch_size'],
                    shuffle=True,
                    num_workers=CONFIG['num_workers'],
                    drop_last=True,
                    pin_memory=True)
    vl = DataLoader(val_ds,
                    batch_size=CONFIG['batch_size'],
                    shuffle=False,
                    num_workers=CONFIG['num_workers'],
                    pin_memory=True)

    # --- MODEL & OPTIMIZER ---
    net = LDE_Model().to(dev)
    loss_fn = LDELoss().to(dev)
    opt = optim.AdamW(
        list(net.parameters()) +
        list(loss_fn.parameters()),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scaler = GradScaler(enabled=CONFIG['mixed_precision'])
    sched = CosineWarmup(opt,
                         warmup_steps=CONFIG['warmup_steps'],
                         t_max=CONFIG['T_max'])

    # --- CHECKPOINT ---
    start_epoch = load_latest_checkpoint(
        CONFIG['output_dir'],
        net, loss_fn, opt, scaler, sched, dev
    )

    # --- LOG SETUP ---
    txt_path = os.path.join(CONFIG['log_dir'],
                            'training_log.txt')
    csv_path = os.path.join(CONFIG['log_dir'],
                            'training_metrics.csv')
    if start_epoch == 0:
        with open(txt_path, 'w') as f:
            f.write('[CONFIG]\n')
            for k, v in CONFIG.items():
                f.write(f"{k}: {v}\n")
            p_millions = sum(p.numel() for p in net.parameters()
                             if p.requires_grad) / 1e6
            f.write(f"trainable_params_M: {p_millions:.2f}\n\n")
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ('epoch','train_loss','val_loss',
                 'l1','ssim','lr','seconds')
            )

    # --- TRAINING LOOP ---
    epoch = start_epoch
    while True:
        epoch += 1
        t0 = time.time()

        # -- Training Phase --
        net.train()
        loss_fn.train()
        train_acc = 0.0
        pbar = tqdm(enumerate(tl),
                    total=len(tl),
                    desc=f"Train E{epoch}")
        for i, (rgb, gt) in pbar:
            rgb, gt = rgb.to(dev), gt.to(dev)
            with autocast(enabled=CONFIG['mixed_precision']):
                out = net(rgb)
                m = loss_fn(out, gt, rgb)
                loss = m['loss'] / CONFIG['accum_steps']
            scaler.scale(loss).backward()
            if (i + 1) % CONFIG['accum_steps'] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), CONFIG['grad_clip'])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            sched.step()
            train_acc += m['loss'].item()
            pbar.set_postfix(L=f"{train_acc/(i+1):.4f}",
                             LR=f"{sched.get_lr()[0]:.1e}")

        train_loss = train_acc / len(tl)

        # -- Validation Phase --
        net.eval()
        loss_fn.eval()
        val_acc = 0.0
        l1_acc = 0.0
        ss_acc = 0.0
        vbar = tqdm(vl, desc=f"Val E{epoch}")
        with torch.inference_mode():
            for rgb, gt in vbar:
                rgb, gt = rgb.to(dev), gt.to(dev)
                out = net(rgb)
                m = loss_fn(out, gt, rgb)
                val_acc += m['loss'].item()
                l1_acc += m['l1'].item()
                ss_acc += m['ssim'].item()
                vbar.set_postfix(L=f"{val_acc/len(vbar):.4f}")

        elapsed = time.time() - t0

        save_checkpoint(epoch, net, loss_fn,
                        opt, scaler, sched,
                        CONFIG['output_dir'])

        # -- Logging --
        val_loss = val_acc / len(vl)
        msg = (f"E{epoch}: Train L {train_loss:.4f}, "
               f"Val L {val_loss:.4f} "
               f"[L1 {l1_acc/len(vl):.4f}, SSIM {ss_acc/len(vl):.4f}] "
               f"({elapsed:.1f}s)")
        print(msg)

        with open(txt_path, 'a') as f:
            f.write(msg + "\n")
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow((
                epoch,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{l1_acc/len(vl):.4f}",
                f"{ss_acc/len(vl):.4f}",
                f"{sched.get_lr()[0]:.3e}",
                round(elapsed, 1)
            ))

if __name__ == '__main__':
    main()
