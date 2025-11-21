import os
import gc
import numpy as np
import scipy.io as sio
import torch
from pathlib import Path
from scipy import signal
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_EXERCISE = "E1"   # 你可以改成 E1 / E2 / E3
ROOT_DATA_DIR = r"F:\A-SCI\Ninapro-DB3"
OUTPUT_DIR = "./datasets/ninapro_db3_amputee"

TARGET_CHANNELS = 8
ORIGINAL_FS = 2000
TARGET_FS = 200
DOWNSAMPLE_FACTOR = ORIGINAL_FS // TARGET_FS

WINDOW_SIZE = 40          # 200ms
STRIDE = 20               # 50% overlap

VAL_REPETITIONS = [6]
TRAIN_REPETITIONS = [1, 2, 3, 4, 5]

EXERCISE_INFO = {
    "E1": {"num_classes": 17},
    "E2": {"num_classes": 23},
    "E3": {"num_classes": 9}
}

# ============================================================================
# UTILITIES
# ============================================================================
def find_mat_files(root_dir, exercise):
    """Find .mat files for selected exercise."""
    root_path = Path(root_dir)
    pattern = f"S*_{exercise}_*.mat"
    return sorted([str(f) for f in root_path.rglob(pattern)])


def downsample_signal(data, factor):
    return signal.decimate(data, factor, axis=0, zero_phase=True)


def normalize_zscore(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def create_windows_center_label(data, labels, window_size, stride):
    """
    Create sliding windows.
    Label = label at the window center (recommended for Ninapro)
    """
    num_samples = len(data)
    windows, window_labels = [], []

    center_offset = window_size // 2

    for start in range(0, num_samples - window_size + 1, stride):
        end = start + window_size

        window = data[start:end]
        center_label = labels[start + center_offset]

        windows.append(window)
        window_labels.append(int(center_label))

    return np.array(windows), np.array(window_labels)


def load_and_preprocess(mat_file):
    """Load one subject file."""
    mat_data = sio.loadmat(mat_file)

    # Extract EMG
    if 'emg' in mat_data:
        emg = mat_data['emg']
    else:
        raise ValueError(f"No EMG in {mat_file}")

    # Labels
    if 'restimulus' in mat_data:
        labels = mat_data['restimulus'].flatten()
    else:
        raise ValueError(f"No labels in {mat_file}")

    # Repetitions
    if 'repetition' in mat_data:
        repetitions = mat_data['repetition'].flatten()
    elif 'rerepetition' in mat_data:
        repetitions = mat_data['rerepetition'].flatten()
    else:
        raise ValueError(f"No repetition info in {mat_file}")

    # Select channels
    emg = emg[:, :TARGET_CHANNELS]

    # Downsample
    emg = downsample_signal(emg, DOWNSAMPLE_FACTOR)
    labels = labels[::DOWNSAMPLE_FACTOR]
    repetitions = repetitions[::DOWNSAMPLE_FACTOR]

    length = min(len(emg), len(labels), len(repetitions))
    emg, labels, repetitions = emg[:length], labels[:length], repetitions[:length]

    # Normalize
    emg = normalize_zscore(emg)

    return emg, labels, repetitions


# ============================================================================
# PROCESS EXERCISE
# ============================================================================
def process_exercise(exercise):
    mat_files = find_mat_files(ROOT_DATA_DIR, exercise)
    if not mat_files:
        print("❌ No .mat found.")
        return

    train_windows_list = []
    train_labels_list = []
    val_windows_list = []
    val_labels_list = []

    print(f"📌 Processing {len(mat_files)} subjects for {exercise}")

    for mat_file in tqdm(mat_files, desc="Subjects"):
        emg, labels, repetitions = load_and_preprocess(mat_file)

        for rep in np.unique(repetitions):
            if rep == 0:
                continue

            mask = repetitions == rep
            rep_emg = emg[mask]
            rep_labels = labels[mask]

            # If repetition has no active labels → skip
            if np.max(rep_labels) == 0:
                continue

            windows, window_labels = create_windows_center_label(
                rep_emg, rep_labels, WINDOW_SIZE, STRIDE
            )

            if rep in VAL_REPETITIONS:
                val_windows_list.append(windows)
                val_labels_list.append(window_labels)
            else:
                train_windows_list.append(windows)
                train_labels_list.append(window_labels)

    # Stack
    train_windows = np.vstack(train_windows_list)
    train_labels = np.concatenate(train_labels_list)
    val_windows = np.vstack(val_windows_list)
    val_labels = np.concatenate(val_labels_list)

    # Remove rest class (0) entirely
    train_mask = train_labels != 0
    val_mask = val_labels != 0

    train_windows = train_windows[train_mask]
    train_labels = train_labels[train_mask]
    val_windows = val_windows[val_mask]
    val_labels = val_labels[val_mask]

    # Build label map
    unique_labels = sorted(list(set(train_labels) | set(val_labels)))
    label_map = {orig: i for i, orig in enumerate(unique_labels)}

    print("\n🔍 Label Mapping:")
    print(label_map)

    # Convert to Waveformer format
    domain_name = exercise.lower()
    full_train, full_val = [], []

    for x, y in zip(train_windows, train_labels):
        sig = torch.from_numpy(x).float().permute(1, 0).unsqueeze(0)
        full_train.append((domain_name, sig, label_map[y]))

    for x, y in zip(val_windows, val_labels):
        sig = torch.from_numpy(x).float().permute(1, 0).unsqueeze(0)
        full_val.append((domain_name, sig, label_map[y]))

    # Save
    out_dir = os.path.join(OUTPUT_DIR, exercise.lower())
    os.makedirs(out_dir, exist_ok=True)
    torch.save(full_train, os.path.join(out_dir, "train.pt"))
    torch.save(full_val, os.path.join(out_dir, "val.pt"))

    print("\n✅ Saved:")
    print(f"Train: {len(full_train)} samples")
    print(f"Val:   {len(full_val)} samples")
    print(f"➡ nb_classes = {len(unique_labels)}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    process_exercise(TARGET_EXERCISE)
