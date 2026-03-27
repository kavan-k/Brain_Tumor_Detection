"""
Visualization module.
Plots sample predictions and overlays.
Contributor: Person A
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from config import OUTPUT_DIR, MODEL_TYPE


def plot_sample_predictions(model, X_test_list, y_test_list, num_samples=5, save=True):
    """
    Display random test samples with FLAIR, ground truth, prediction, and overlay.
    2D mode only; 3D would require a slice selection strategy.
    """
    if MODEL_TYPE == '3d':
        print("3D visualization not implemented in this version.")
        return

    device = next(model.parameters()).device
    model.eval()

    indices = np.random.choice(len(X_test_list), num_samples, replace=False)

    for plot_i, idx in enumerate(indices):
        image = X_test_list[idx].astype(np.float32)   # (H, W, 4)
        mask  = y_test_list[idx].astype(np.float32)   # (H, W)

        # Run inference
        img_t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)  # (1,4,H,W)
        with torch.no_grad():
            pred = model(img_t)[0, 0].cpu().numpy()   # (H, W)

        pred_bin = (pred > 0.5).astype(np.float32)

        # Use FLAIR (channel index 3) as the background grayscale image
        flair = image[:, :, 3]

        # Normalise FLAIR to [0, 1] for overlay display
        flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

        # Overlay: FLAIR in grayscale, predicted mask in red
        overlay = np.stack([flair_norm] * 3, axis=-1)
        overlay[:, :, 0] = np.where(pred_bin == 1, 1.0, flair_norm)
        overlay[:, :, 1] = np.where(pred_bin == 1, 0.0, flair_norm)
        overlay[:, :, 2] = np.where(pred_bin == 1, 0.0, flair_norm)
        overlay = np.clip(overlay, 0.0, 1.0)

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(flair_norm, cmap='gray'); axes[0].set_title('FLAIR');        axes[0].axis('off')
        axes[1].imshow(mask,       cmap='gray'); axes[1].set_title('Ground Truth'); axes[1].axis('off')
        axes[2].imshow(pred_bin,   cmap='gray'); axes[2].set_title('Prediction');   axes[2].axis('off')
        axes[3].imshow(overlay);                 axes[3].set_title('Overlay');      axes[3].axis('off')

        plt.tight_layout()
        if save:
            path = os.path.join(OUTPUT_DIR, f'pred_sample_{plot_i}.png')
            plt.savefig(path, dpi=150)
            print(f"  Saved: {path}")
        plt.show()