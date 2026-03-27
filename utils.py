"""
Utility functions: custom loss, metrics, etc.
Contributor: Person A (shared)
"""

import torch


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for binary segmentation. Inputs are torch tensors."""
    y_true = y_true.float()
    y_pred = y_pred.float()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss = 1 - dice coefficient."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def iou(y_true, y_pred, smooth=1e-6):
    """Intersection over Union (Jaccard index)."""
    y_true = y_true.float()
    y_pred = y_pred.float()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    """
    BCE + Dice loss.
    More stable than pure Dice on class-imbalanced data (e.g. sparse tumor masks).
    """
    bce = torch.nn.functional.binary_cross_entropy(y_pred, y_true.float())
    return bce + dice_loss(y_true, y_pred)


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """Tversky loss — penalises false negatives more than false positives."""
    y_true = y_true.float()
    y_pred = y_pred.float()
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky