"""
Preprocessing module.
Handles normalization, resizing, and data augmentation.
Contributor: Person B

Augmentation is done with torchvision transforms instead of Keras ImageDataGenerator.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
import random
from skimage.transform import resize
from config import IMG_SIZE, USE_AUGMENTATION


def normalize_volume(volume):
    """Zero-mean unit-variance normalization per volume."""
    return (volume - np.mean(volume)) / (np.std(volume) + 1e-8)


def preprocess_2d_sample(sample_4ch, mask):
    """Resize a 2D sample and its mask to target IMG_SIZE."""
    sample_resized = resize(sample_4ch, IMG_SIZE + (4,), preserve_range=True, anti_aliasing=True)
    mask_resized = resize(mask, IMG_SIZE, preserve_range=True, order=0, anti_aliasing=False)
    mask_resized = (mask_resized > 0.5).astype(np.float32)
    return sample_resized, mask_resized


def preprocess_3d_patch(patch, mask):
    """For 3D patches, no resizing is applied."""
    return patch.astype(np.float32), mask.astype(np.float32)


def augment_2d_sample(image, mask):
    """
    Apply consistent random augmentation to a single (H, W, C) image and (H, W) mask.
    Both receive identical random transforms so they stay aligned.

    Uses torchvision functional transforms — no interpolation on the binary mask.
    """
    # Convert to tensors: image → (C, H, W), mask → (1, H, W)
    img_t  = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))  # (4, H, W)
    mask_t = torch.from_numpy(mask[np.newaxis].astype(np.float32))           # (1, H, W)

    # Random horizontal flip
    if random.random() > 0.5:
        img_t  = TF.hflip(img_t)
        mask_t = TF.hflip(mask_t)

    # Random rotation ±10°
    angle = random.uniform(-10, 10)
    img_t  = TF.rotate(img_t,  angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    mask_t = TF.rotate(mask_t, angle, interpolation=TF.InterpolationMode.NEAREST,  fill=0)

    # Random translate ±5 % of image size
    h, w = img_t.shape[-2], img_t.shape[-1]
    max_dx, max_dy = int(0.05 * w), int(0.05 * h)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    img_t  = TF.affine(img_t,  angle=0, translate=[dx, dy], scale=1.0, shear=0,
                       interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    mask_t = TF.affine(mask_t, angle=0, translate=[dx, dy], scale=1.0, shear=0,
                       interpolation=TF.InterpolationMode.NEAREST,  fill=0)

    # Random zoom 0.95 – 1.05 (centre-crop then resize back)
    scale = random.uniform(0.95, 1.05)
    new_h, new_w = int(h * scale), int(w * scale)
    img_t  = TF.resize(img_t,  [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    mask_t = TF.resize(mask_t, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST,  antialias=False)
    img_t  = TF.center_crop(img_t,  [h, w])
    mask_t = TF.center_crop(mask_t, [h, w])

    # Re-binarize mask (nearest-neighbour should be clean, but ensure it)
    mask_t = (mask_t > 0.5).float()

    # Back to numpy: image → (H, W, C), mask → (H, W)
    image_aug = img_t.numpy().transpose(1, 2, 0)
    mask_aug  = mask_t.numpy().squeeze(0)
    return image_aug, mask_aug