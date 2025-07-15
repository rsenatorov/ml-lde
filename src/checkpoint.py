# src/checkpoint.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
Helper functions for saving and loading LDE checkpoints.
"""
import os
import re
import torch

CHECKPOINT_PREFIX = "lde_epoch"

def save_checkpoint(epoch, model, loss_fn, opt, scaler, sched, out_dir):
    """
    Save training state: model, optimizer, loss, scaler, scheduler.
    """
    os.makedirs(out_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'net': model.state_dict(),
        'loss': loss_fn.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'sched': sched.state_dict()
    }
    path = os.path.join(out_dir,
                        f"{CHECKPOINT_PREFIX}{epoch}.pth")
    torch.save(state, path)
    print(f"[INFO] Checkpoint saved to {path}")

def load_latest_checkpoint(out_dir, model, loss_fn,
                           opt, scaler, sched, device):
    """
    Load the most recent checkpoint if available.
    Returns the last epoch number (0 if none).
    """
    if not os.path.isdir(out_dir):
        return 0
    files = [f for f in os.listdir(out_dir)
             if f.startswith(CHECKPOINT_PREFIX)
             and f.endswith('.pth')]
    if not files:
        print("[INFO] No checkpoints found. Starting fresh.")
        return 0
    epochs = [int(re.findall(r'(\d+)', f)[0]) for f in files]
    latest = max(epochs)
    path = os.path.join(out_dir,
                        f"{CHECKPOINT_PREFIX}{latest}.pth")
    print(f"[INFO] Loading checkpoint {path}")
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck['net'])
    loss_fn.load_state_dict(ck['loss'])
    opt.load_state_dict(ck['opt'])
    scaler.load_state_dict(ck['scaler'])
    sched.load_state_dict(ck['sched'])
    print(f"[INFO] Resumed from epoch {latest + 1}")
    return latest
