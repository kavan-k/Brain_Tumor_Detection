"""
Evaluation module.
Computes metrics on the test set and prints results.
Contributor: Person C
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import MODEL_TYPE, BATCH_SIZE
from train import BraTSDataset
from utils import combined_loss, dice_coefficient, iou


def evaluate_model(model, X_test_list, y_test_list):
    """
    Evaluate model on test data.
    Returns (avg_loss, avg_dice, avg_iou).
    """
    device = next(model.parameters()).device
    model.eval()

    test_ds     = BraTSDataset(X_test_list, y_test_list, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=(device.type == 'cuda'))

    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    dice_per_sample, iou_per_sample   = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds   = model(X_batch)
            preds_bin = (preds > 0.5).float()

            total_loss += combined_loss(y_batch, preds).item()
            total_dice += dice_coefficient(y_batch, preds_bin).item()
            total_iou  += iou(y_batch, preds_bin).item()

            # Per-sample metrics
            for i in range(len(y_batch)):
                d = dice_coefficient(y_batch[i], preds_bin[i]).item()
                s = iou(y_batch[i], preds_bin[i]).item()
                dice_per_sample.append(d)
                iou_per_sample.append(s)

    n = len(test_loader)
    avg_loss = total_loss / n
    avg_dice = total_dice / n
    avg_iou  = total_iou  / n

    print(f"\nTest Results:")
    print(f"  Loss : {avg_loss:.4f}")
    print(f"  Dice : {avg_dice:.4f}")
    print(f"  IoU  : {avg_iou:.4f}")
    print(f"  Per-sample Dice: {np.mean(dice_per_sample):.4f} ± {np.std(dice_per_sample):.4f}")
    print(f"  Per-sample IoU : {np.mean(iou_per_sample):.4f} ± {np.std(iou_per_sample):.4f}")

    return avg_loss, avg_dice, avg_iou