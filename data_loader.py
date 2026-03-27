"""
Data loading module.
Responsible for downloading the dataset (if not present) and loading patient scans.
Contributor: Person A
"""

import os
import zipfile
import subprocess
import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize
from config import DATA_DIR, KAGGLE_DATASET, MODALITIES, FILTER_EMPTY_SLICES, IMG_SIZE

def download_brats():
    """Download BraTS dataset using Kaggle API."""
    if os.path.exists(os.path.join(DATA_DIR, "BraTS2020_TrainingData")):
        print("Dataset already downloaded.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading BraTS 2020 dataset from Kaggle...")
    try:
        cmd = f"kaggle datasets download {KAGGLE_DATASET} -p {DATA_DIR}"
        subprocess.run(cmd, shell=True, check=True)
        print("Download complete.")
    except subprocess.CalledProcessError:
        print("Kaggle API failed. Please ensure you have installed kaggle and placed your API token.")
        raise

    zip_path = os.path.join(DATA_DIR, "brats20-dataset-training-validation.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path)
        print("Extraction complete.")
    else:
        print("ZIP file not found. Please check download.")
        raise FileNotFoundError("Dataset ZIP not found.")

def is_patient_folder(folder_path):
    """Check if a folder contains at least one modality file."""
    files = os.listdir(folder_path)
    for mod in MODALITIES:
        for f in files:
            if f.endswith(f'_{mod}.nii') or f.endswith(f'_{mod}.nii.gz') or f == f'{mod}.nii' or f == f'{mod}.nii.gz':
                return True
    return False

def find_patient_folders(root_dir):
    """Recursively find all patient folders."""
    patient_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if is_patient_folder(dirpath):
            patient_folders.append(dirpath)
            dirnames.clear()  # don't descend further
    return patient_folders

def load_patient_data(patient_path):
    """Load all modalities and segmentation for one patient."""
    data = {}
    base = os.path.basename(patient_path)

    for mod in MODALITIES:
        possible_paths = [
            os.path.join(patient_path, f'{base}_{mod}.nii'),
            os.path.join(patient_path, f'{base}_{mod}.nii.gz'),
            os.path.join(patient_path, f'{mod}.nii'),
            os.path.join(patient_path, f'{mod}.nii.gz'),
            os.path.join(patient_path, f'BraTS20_Training_{base}_{mod}.nii'),
            os.path.join(patient_path, f'BraTS20_Training_{base}_{mod}.nii.gz')
        ]
        loaded = False
        for p in possible_paths:
            if os.path.exists(p):
                data[mod] = nib.load(p).get_fdata().astype(np.float32)
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(f"Missing modality {mod} in {patient_path} (tried all variants)")

    seg_paths = [
        os.path.join(patient_path, f'{base}_seg.nii'),
        os.path.join(patient_path, f'{base}_seg.nii.gz'),
        os.path.join(patient_path, 'seg.nii'),
        os.path.join(patient_path, 'seg.nii.gz'),
        os.path.join(patient_path, f'BraTS20_Training_{base}_seg.nii'),
        os.path.join(patient_path, f'BraTS20_Training_{base}_seg.nii.gz')
    ]
    loaded = False
    for p in seg_paths:
        if os.path.exists(p):
            data['seg'] = nib.load(p).get_fdata().astype(np.float32)
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError(f"Missing segmentation in {patient_path}")

    return data

def normalize_volume(volume):
    """Zero-mean unit-variance normalization."""
    return (volume - np.mean(volume)) / (np.std(volume) + 1e-8)

def extract_slices(patient_data):
    """
    Extract 2D axial slices, resize to IMG_SIZE, and convert to float16.
    Returns:
        X_slices: list of (H, W, 4) float16 arrays
        y_slices: list of (H, W) float32 binary masks
    """
    stacked = np.stack([patient_data[mod] for mod in MODALITIES], axis=-1)  # (240,240,155,4)
    seg = patient_data['seg']
    seg = (seg > 0).astype(np.float32)

    X_slices, y_slices = [], []
    depth = stacked.shape[2]
    for slice_idx in range(depth):
        slice_4ch = stacked[:, :, slice_idx, :]  # (240,240,4)
        slice_mask = seg[:, :, slice_idx]        # (240,240)

        # Resize
        slice_resized = resize(slice_4ch, IMG_SIZE + (4,), preserve_range=True, anti_aliasing=True).astype(np.float16)
        mask_resized = resize(slice_mask, IMG_SIZE, preserve_range=True, order=0, anti_aliasing=False).astype(np.float32)
        mask_resized = (mask_resized > 0.5).astype(np.float32)

        if FILTER_EMPTY_SLICES and np.sum(mask_resized) == 0:
            continue

        X_slices.append(slice_resized)
        y_slices.append(mask_resized)

    return X_slices, y_slices

def extract_patches_3d(patient_data, patch_size):
    """Extract 3D patches (not used in 2D mode, but kept for completeness)."""
    stacked = np.stack([patient_data[mod] for mod in MODALITIES], axis=-1)
    seg = (patient_data['seg'] > 0).astype(np.float32)

    d, h, w = patch_size
    D, H, W, _ = stacked.shape
    X_patches, y_patches = [], []
    for i in range(0, D, d):
        for j in range(0, H, h):
            for k in range(0, W, w):
                if i+d <= D and j+h <= H and k+w <= W:
                    patch_X = stacked[i:i+d, j:j+h, k:k+w, :].astype(np.float16)
                    patch_y = seg[i:i+d, j:j+h, k:k+w].astype(np.float32)
                    X_patches.append(patch_X)
                    y_patches.append(patch_y)
    return X_patches, y_patches

def load_all_patients(data_root, model_type='2d', patch_size=None):
    """
    Load all patients and return lists of samples and masks.
    Returns:
        X_list: list of samples (float16)
        y_list: list of masks (float32)
    """
    X_list, y_list = [], []
    possible_roots = [
        os.path.join(data_root, "BraTS2020_TrainingData"),
        os.path.join(data_root, "MICCAI_BraTS2020_TrainingData"),
        data_root
    ]
    root_dir = None
    for r in possible_roots:
        if os.path.isdir(r):
            root_dir = r
            print(f"Using root directory: {root_dir}")
            break
    if root_dir is None:
        raise FileNotFoundError(f"No training data directory found in {data_root}")

    print("Scanning for patient folders...")
    patient_folders = find_patient_folders(root_dir)
    print(f"Found {len(patient_folders)} patient folders.")

    patient_count = 0
    for patient_path in tqdm(patient_folders, desc="Loading patients"):
        try:
            data = load_patient_data(patient_path)
            for mod in MODALITIES:
                data[mod] = normalize_volume(data[mod])
            if model_type == '2d':
                X_slices, y_slices = extract_slices(data)
                X_list.extend(X_slices)
                y_list.extend(y_slices)
                patient_count += 1
            elif model_type == '3d':
                if patch_size is None:
                    raise ValueError("patch_size required for 3D")
                X_patches, y_patches = extract_patches_3d(data, patch_size)
                X_list.extend(X_patches)
                y_list.extend(y_patches)
                patient_count += 1
        except Exception as e:
            print(f"Error loading {patient_path}: {e}")
            continue

    print(f"Successfully loaded {patient_count} patients.")
    print(f"Total samples: {len(X_list)}")
    if len(X_list) == 0:
        print("WARNING: No samples loaded. Check dataset structure and patient files.")
    return X_list, y_list