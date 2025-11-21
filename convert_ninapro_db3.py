"""
Ninapro DB3 Preprocessing Script for WaveFormer
Simulates 8-channel armband at 200Hz for transradial amputee research
"""

import os
import gc
import numpy as np
import scipy.io as sio
import torch
from pathlib import Path
from scipy import signal
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - Change these variables as needed
# ============================================================================
TARGET_EXERCISE = "E1"  # Options: "E1", "E2", "E3"
ROOT_DATA_DIR = r"F:\A-SCI\Ninapro-DB3"
OUTPUT_DIR = "./datasets/ninapro_db3_amputee"

# Hardware simulation parameters
TARGET_CHANNELS = 8  # Simulate 8-channel armband (use channels 0-7)
ORIGINAL_FS = 2000  # Ninapro DB3 sampling rate
TARGET_FS = 200  # Target sampling rate
DOWNSAMPLE_FACTOR = ORIGINAL_FS // TARGET_FS  # Factor of 10

# Windowing parameters
WINDOW_SIZE = 40  # 200ms at 200Hz
STRIDE = 20  # 50% overlap

# Data splitting strategy
VAL_REPETITIONS = [2, 5]  # Reps 2 and 5 for validation
TRAIN_REPETITIONS = [1, 3, 4, 6]  # Others for training

# Exercise configurations
EXERCISE_INFO = {
    "E1": {"num_classes": 17, "description": "B"},
    "E2": {"num_classes": 23, "description": "C"},
    "E3": {"num_classes": 9, "description": "D"}
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_mat_files(root_dir, exercise):
    """Recursively find all .mat files for the specified exercise."""
    mat_files = []
    root_path = Path(root_dir)

    # Search pattern for exercise files
    pattern = f"S*_{exercise}_*.mat"

    for mat_file in root_path.rglob(pattern):
        mat_files.append(str(mat_file))

    if not mat_files:
        print(f"Warning: No .mat files found for {exercise} in {root_dir}")

    return sorted(mat_files)


def downsample_signal(data, factor):
    """Downsample signal using decimation (anti-aliasing filter included)."""
    return signal.decimate(data, factor, axis=0, zero_phase=True)


def normalize_zscore(data):
    """Apply Z-score normalization per channel."""
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std[std == 0] = 1.0  # Avoid division by zero
    return (data - mean) / std


def create_windows(data, labels, window_size, stride):
    """
    Create sliding windows from continuous data.

    Returns:
        windows: (num_windows, window_size, num_channels)
        window_labels: (num_windows,)
    """
    num_samples = len(data)
    windows = []
    window_labels = []

    for start in range(0, num_samples - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]

        # Use majority voting for label (most common label in window)
        label_counts = np.bincount(labels[start:end].astype(int))
        majority_label = np.argmax(label_counts)

        windows.append(window)
        window_labels.append(majority_label)

    return np.array(windows), np.array(window_labels)


def load_and_preprocess_subject(mat_file, target_channels, downsample_factor):
    """
    Load a single subject's .mat file and preprocess it.

    Returns:
        emg_data: Preprocessed EMG data (downsampled, channel-selected)
        labels: Corresponding labels
        repetitions: Repetition indices
    """
    try:
        mat_data = sio.loadmat(mat_file)

        # Extract EMG data (assuming field name 'emg')
        if 'emg' in mat_data:
            emg = mat_data['emg']
        else:
            # Try alternative field names
            possible_fields = ['data', 'signal', 'EMG']
            emg = None
            for field in possible_fields:
                if field in mat_data:
                    emg = mat_data[field]
                    break

            if emg is None:
                raise ValueError(f"Could not find EMG data in {mat_file}")

        # Extract labels (stimulus/restimulus)
        if 'restimulus' in mat_data:
            labels = mat_data['restimulus'].flatten()
        elif 'stimulus' in mat_data:
            labels = mat_data['stimulus'].flatten()
        else:
            raise ValueError(f"Could not find labels in {mat_file}")

        # Extract repetitions
        if 'rerepetition' in mat_data:
            repetitions = mat_data['rerepetition'].flatten()
        elif 'repetition' in mat_data:
            repetitions = mat_data['repetition'].flatten()
        else:
            raise ValueError(f"Could not find repetition info in {mat_file}")

        # Select first 8 channels (0-7)
        emg = emg[:, :target_channels]

        # Downsample from 2000Hz to 200Hz
        emg = downsample_signal(emg, downsample_factor)
        labels = labels[::downsample_factor]
        repetitions = repetitions[::downsample_factor]

        # Ensure all arrays have same length
        min_len = min(len(emg), len(labels), len(repetitions))
        emg = emg[:min_len]
        labels = labels[:min_len]
        repetitions = repetitions[:min_len]

        return emg, labels, repetitions

    except Exception as e:
        print(f"Error loading {mat_file}: {e}")
        return None, None, None


def process_exercise(exercise, root_dir, output_dir):
    """
    Process all subjects for a given exercise.
    """
    print(f"\n{'=' * 70}")
    print(f"Processing Exercise: {exercise}")
    print(f"Description: {EXERCISE_INFO[exercise]['description']}")
    print(
        f"Number of classes: {EXERCISE_INFO[exercise]['num_classes']} + 1 (rest) = {EXERCISE_INFO[exercise]['num_classes'] + 1}")
    print(f"{'=' * 70}\n")

    # Find all .mat files for this exercise
    mat_files = find_mat_files(root_dir, exercise)

    if not mat_files:
        print(f"ERROR: No .mat files found for {exercise}. Exiting.")
        return

    print(f"Found {len(mat_files)} subject files")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Storage for train and validation data
    train_windows_list = []
    train_labels_list = []
    val_windows_list = []
    val_labels_list = []

    # Process each subject
    for mat_file in tqdm(mat_files, desc="Processing subjects"):
        subject_name = Path(mat_file).stem

        # Load and preprocess
        emg, labels, repetitions = load_and_preprocess_subject(
            mat_file, TARGET_CHANNELS, DOWNSAMPLE_FACTOR
        )

        if emg is None:
            continue

        # Normalize
        emg = normalize_zscore(emg)

        # Split by repetition
        for rep in np.unique(repetitions):
            rep = int(rep)
            if rep == 0:  # Skip rest periods between repetitions
                continue

            # Get data for this repetition
            rep_mask = repetitions == rep
            rep_emg = emg[rep_mask]
            rep_labels = labels[rep_mask]

            # Create windows
            windows, window_labels = create_windows(
                rep_emg, rep_labels, WINDOW_SIZE, STRIDE
            )

            # Assign to train or validation
            if rep in VAL_REPETITIONS:
                val_windows_list.append(windows)
                val_labels_list.append(window_labels)
            elif rep in TRAIN_REPETITIONS:
                train_windows_list.append(windows)
                train_labels_list.append(window_labels)

        # Clean up memory
        del emg, labels, repetitions
        gc.collect()

    print("\nStacking and saving data in WaveFormer official format...")

    domain_name = TARGET_EXERCISE.lower()

    # 1. 合并所有原始窗口和标签（暂不区分训练/验证，先统一计算映射）
    # 注意：这里我们先把 train 和 val 的列表合并处理逻辑

    # 收集所有 Train 的数据
    all_train_windows = np.vstack(train_windows_list) if train_windows_list else np.empty(
        (0, WINDOW_SIZE, TARGET_CHANNELS))
    all_train_labels = np.concatenate(train_labels_list) if train_labels_list else np.array([])

    # 收集所有 Val 的数据
    all_val_windows = np.vstack(val_windows_list) if val_windows_list else np.empty((0, WINDOW_SIZE, TARGET_CHANNELS))
    all_val_labels = np.concatenate(val_labels_list) if val_labels_list else np.array([])

    # 2. 自动计算标签映射 (Label Mapping)
    # 找出数据中存在的所有唯一标签（排除 0 休息）
    unique_labels = sorted(list(set(np.unique(all_train_labels)) | set(np.unique(all_val_labels))))
    if 0 in unique_labels:
        unique_labels.remove(0)  # 移除休息类

    # 创建映射字典：{原始标签: 新标签(0~N-1)}
    label_map = {original: new_idx for new_idx, original in enumerate(unique_labels)}

    print(f"\n🔄 标签自动重映射检测:")
    print(f"   原始标签范围: {unique_labels[0]} - {unique_labels[-1]}")
    print(f"   检测到类别数: {len(unique_labels)}")
    print(
        f"   映射示例: {unique_labels[0]}->0, {unique_labels[1]}->1 ... {unique_labels[-1]}->{len(unique_labels) - 1}")

    # 3. 构建并转换训练集 (Train)
    full_train_list = []
    print(f"Formatting {len(all_train_windows)} training windows...", end=" ")
    for i in tqdm(range(len(all_train_windows)), desc="Train", leave=False):
        orig_label = int(all_train_labels[i])
        if orig_label == 0: continue  # 跳过休息

        # 使用字典映射标签
        if orig_label not in label_map: continue  # 安全检查
        new_label = label_map[orig_label]

        signal = all_train_windows[i]
        tensor_signal = torch.from_numpy(signal).float().permute(1, 0).unsqueeze(0)
        full_train_list.append((domain_name, tensor_signal, new_label))
    print(f"Done → {len(full_train_list)} samples")

    # 4. 构建并转换验证集 (Val)
    full_val_list = []
    print(f"Formatting {len(all_val_windows)} validation windows...", end=" ")
    for i in tqdm(range(len(all_val_windows)), desc="Val", leave=False):
        orig_label = int(all_val_labels[i])
        if orig_label == 0: continue  # 跳过休息

        # 使用字典映射标签
        if orig_label not in label_map: continue
        new_label = label_map[orig_label]

        signal = all_val_windows[i]
        tensor_signal = torch.from_numpy(signal).float().permute(1, 0).unsqueeze(0)
        full_val_list.append((domain_name, tensor_signal, new_label))
    print(f"Done → {len(full_val_list)} samples")

    # 5. 保存
    os.makedirs(exercise_output_dir, exist_ok=True)
    train_path = os.path.join(exercise_output_dir, "train.pt")
    val_path = os.path.join(exercise_output_dir, "val.pt")

    torch.save(full_train_list, train_path)
    torch.save(full_val_list, val_path)

    print(f"\n✅ Data Saved Successfully!")
    print(f"   Train: {train_path}")
    print(f"   Val:   {val_path}")
    print(f"⚠️ 训练时请务必设置: --nb_classes {len(unique_labels)}")
    print(f"{'=' * 70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║          Ninapro DB3 Preprocessing for WaveFormer                ║
    ║              8-Channel Armband Simulation @ 200Hz                ║
    ╚══════════════════════════════════════════════════════════════════╝

    Configuration:
    - Target Exercise: {TARGET_EXERCISE}
    - Root Directory: {ROOT_DATA_DIR}
    - Output Directory: {OUTPUT_DIR}
    - Channels: {TARGET_CHANNELS} (0-7)
    - Sampling Rate: {ORIGINAL_FS}Hz → {TARGET_FS}Hz
    - Window: {WINDOW_SIZE} samples ({WINDOW_SIZE / TARGET_FS * 1000:.0f}ms)
    - Stride: {STRIDE} samples ({STRIDE / TARGET_FS * 1000:.0f}ms overlap)
    """)

    # Validate exercise selection
    if TARGET_EXERCISE not in EXERCISE_INFO:
        print(f"ERROR: Invalid exercise '{TARGET_EXERCISE}'. Must be one of: {list(EXERCISE_INFO.keys())}")
        exit(1)

    # Create output directory with exercise name
    exercise_output_dir = os.path.join(OUTPUT_DIR, TARGET_EXERCISE.lower())

    # Process the selected exercise
    process_exercise(TARGET_EXERCISE, ROOT_DATA_DIR, exercise_output_dir)

    print("\n✓ All done! Ready for training with WaveFormer.")
    print(f"\nNext step: Run training with:")
    print(f"  --data_path {exercise_output_dir}")
    print(f"  --nb_classes {EXERCISE_INFO[TARGET_EXERCISE]['num_classes'] + 1}")