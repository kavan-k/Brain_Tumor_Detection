"""
Configuration file for brain tumor segmentation project.
All hyperparameters and paths are stored here.
"""

import os

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "brats_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
KAGGLE_DATASET = "awsaf49/brats20-dataset-training-validation"

# -------------------- Data parameters --------------------
IMG_SIZE = (128, 128)                      # Resize slices to this size (height, width)
MODALITIES = ['t1', 't1ce', 't2', 'flair']
FILTER_EMPTY_SLICES = True                 # Skip slices with no tumor (reduces data)

# -------------------- Training parameters --------------------
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# -------------------- Model parameters --------------------
MODEL_TYPE = '2d'                          # '2d' or '3d'
PATCH_SIZE = (32, 128, 128)                # For 3D U-Net (depth, height, width)

# -------------------- Augmentation parameters --------------------
USE_AUGMENTATION = False  # Set to True to enable data augmentation during training