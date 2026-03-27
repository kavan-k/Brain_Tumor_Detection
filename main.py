"""
Main orchestration script.
Runs the entire pipeline: download, preprocess, train, evaluate, visualize.
Contributor: All
"""

import os
import random
import numpy as np
import torch
import multiprocessing
multiprocessing.freeze_support()
from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_TYPE, PATCH_SIZE, RANDOM_SEED,
    VALIDATION_SPLIT, TEST_SPLIT
)
from data_loader import download_brats, load_all_patients
from train import train_model
from evaluate import evaluate_model
from visualize import plot_sample_predictions

CACHE_PATH = os.path.join(DATA_DIR, "cache_2d.npz")

def load_or_cache(data_dir, model_type, patch_size):
    if os.path.exists(CACHE_PATH):
        print("Loading cached data...")
        data = np.load(CACHE_PATH, allow_pickle=True)
        return list(data['X']), list(data['y'])
    else:
        print("No cache found. Loading from raw files (one-time)...")
        X_list, y_list = load_all_patients(data_dir, model_type=model_type, patch_size=patch_size)
        print("Saving cache for future runs...")
        np.savez_compressed(CACHE_PATH, X=np.array(X_list), y=np.array(y_list))
        print(f"Cache saved to {CACHE_PATH}")
        return X_list, y_list

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seeds(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # GPU check 
    if torch.cuda.is_available():
        print(f"CUDA available — {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available — running on CPU (trainingwill be slow)")

    # Step 1: Download dataset
    download_brats()

    # Step 2: Load data
    print("\nLoading patient data")
    X_list, y_list = load_all_patients(
        DATA_DIR,
        model_type=MODEL_TYPE,
        patch_size=PATCH_SIZE if MODEL_TYPE == '3d' else None
    )
    print(f"Total samples: {len(X_list)}")

    if len(X_list) == 0:
        print("\nNo data loaded. Exiting.")
        return

    # Step 3: Split into train/val/test sets
    print("\nSplitting data into train/val/test")
    num_samples = len(X_list)
    indices     = np.random.permutation(num_samples)

    test_count  = int(num_samples * TEST_SPLIT)
    val_count   = int(num_samples * VALIDATION_SPLIT)

    test_idx  = indices[:test_count]
    val_idx   = indices[test_count:test_count + val_count]
    train_idx = indices[test_count + val_count:]

    X_train = [X_list[i] for i in train_idx]
    y_train = [y_list[i] for i in train_idx]
    X_val   = [X_list[i] for i in val_idx]
    y_val   = [y_list[i] for i in val_idx]
    X_test  = [X_list[i] for i in test_idx]
    y_test  = [y_list[i] for i in test_idx]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    if len(X_train) == 0:
        print("No training samples. Exiting.")
        return

    # Step 4: Train
    print("\nTraining model")
    model, history = train_model(X_train, y_train, X_val, y_val)

    # Step 5: Evaluate
    print("\n=== Evaluating on test set ===")
    evaluate_model(model, X_test, y_test)

    # Step 6: Visualize predictions 
    print("\nGenerating sample predictions for visualization")
    plot_sample_predictions(model, X_test, y_test, num_samples=5)

    print(f"\nAll done. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()