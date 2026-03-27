"""
Training module.
Handles model compilation, training loop, callbacks, and saving history.
Contributor: Person B
"""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_TYPE, USE_AUGMENTATION
from model import get_model
from utils import combined_loss, dice_coefficient, iou
from preprocessing import augment_2d_sample


# Dataset

class BraTSDataset(Dataset):
    """
    PyTorch Dataset that wraps lists of numpy arrays.

    Images are stored as (H, W, 4) float16/float32 numpy arrays.
    Masks  are stored as (H, W) float32 numpy arrays.

    __getitem__ converts to channel-first float32 tensors:
      image → (4, H, W)
      mask  → (1, H, W)
    """
    def __init__(self, X_list, y_list, augment=False):
    # Pre-cast to float32 once to avoid per-sample conversion during training
        self.X = [x.astype(np.float32) for x in X_list]
        self.y = [y.astype(np.float32) for y in y_list]
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx].astype(np.float32)   # (H, W, 4)
        mask  = self.y[idx].astype(np.float32)   # (H, W)

        if self.augment and USE_AUGMENTATION and MODEL_TYPE == '2d':
            image, mask = augment_2d_sample(image, mask)

        # Convert to channel-first tensors
        image_t = torch.from_numpy(image.transpose(2, 0, 1))  # (4, H, W)
        mask_t  = torch.from_numpy(mask[np.newaxis])           # (1, H, W)
        return image_t, mask_t


# Training function

torch.backends.cudnn.benchmark = True  # optimises conv kernels for fixed input size
def train_model(X_train_list, y_train_list, X_val_list, y_val_list):
    """Train the U-Net model and return (model, history_dict)."""

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Datasets & DataLoaders
    train_ds = BraTSDataset(X_train_list, y_train_list, augment=True)
    val_ds   = BraTSDataset(X_val_list,   y_val_list,   augment=False)

    train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True,
    persistent_workers=True
    )
    val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True,
    persistent_workers=True
    )

    print(f"Training samples  : {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Batch size        : {BATCH_SIZE}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = get_model(MODEL_TYPE).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    # ── History ──────────────────────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou':  [], 'val_iou':  [],
    }

    best_val_dice = -1.0
    patience_counter = 0
    early_stop_patience = 15
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        t_loss, t_dice, t_iou = 0.0, 0.0, 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = combined_loss(y_batch, preds)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                t_loss += loss.item()
                t_dice += dice_coefficient(y_batch, (preds > 0.5).float()).item()
                t_iou  += iou(y_batch, (preds > 0.5).float()).item()

        n_train = len(train_loader)
        t_loss /= n_train; t_dice /= n_train; t_iou /= n_train

        # ── Validate ──
        model.eval()
        v_loss, v_dice, v_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds   = model(X_batch)
                v_loss += combined_loss(y_batch, preds).item()
                v_dice += dice_coefficient(y_batch, (preds > 0.5).float()).item()
                v_iou  += iou(y_batch, (preds > 0.5).float()).item()

        n_val = len(val_loader)
        v_loss /= n_val; v_dice /= n_val; v_iou /= n_val

        scheduler.step(v_loss)

        # ── Log ──
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_dice'].append(t_dice)
        history['val_dice'].append(v_dice)
        history['train_iou'].append(t_iou)
        history['val_iou'].append(v_iou)

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Loss {t_loss:.4f}/{v_loss:.4f} | "
            f"Dice {t_dice:.4f}/{v_dice:.4f} | "
            f"IoU {t_iou:.4f}/{v_iou:.4f} | "
            f"LR {lr_now:.2e}"
        )

        # ── Checkpoint ──
        if v_dice > best_val_dice:
            best_val_dice = v_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved (val_dice={best_val_dice:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ── Load best weights ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model weights from {best_model_path}")

    # ── Training curves ──────────────────────────────────────────────────────
    epochs_ran = len(history['train_loss'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss (BCE + Dice)')
    axes[0].set_xlabel('Epoch'); axes[0].legend()

    axes[1].plot(history['train_dice'], label='Train')
    axes[1].plot(history['val_dice'],   label='Val')
    axes[1].set_title('Dice Coefficient')
    axes[1].set_xlabel('Epoch'); axes[1].legend()

    axes[2].plot(history['train_iou'], label='Train')
    axes[2].plot(history['val_iou'],   label='Val')
    axes[2].set_title('IoU')
    axes[2].set_xlabel('Epoch'); axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    plt.show()

    return model, history