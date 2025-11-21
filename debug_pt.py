import torch
import os
from collections import Counter


def inspect_dataset(file_path):
    print(f"\n{'=' * 20} Inspecting: {file_path} {'=' * 20}")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    try:
        # 加载数据
        data = torch.load(file_path)
        print(f"✅ Data loaded. Total samples: {len(data)}")

        if len(data) == 0:
            print("⚠️ Dataset is empty!")
            return

        # 提取标签 (根据 WaveFormer 格式，数据是 (domain, signal, label) 的元组)
        # label 是第三个元素 (index 2)
        labels = [int(item[2]) for item in data]

        label_counts = Counter(labels)
        sorted_labels = sorted(label_counts.items())

        print(f"🔢 Unique classes found: {len(sorted_labels)}")
        print(f"📊 Class Distribution:")
        for label, count in sorted_labels:
            print(f"   Class {label}: {count} samples")

        if len(sorted_labels) <= 1:
            print("\n🚨 CRITICAL ISSUE: Dataset contains only 1 class! This explains the 100% accuracy.")
        else:
            print("\n✅ Class distribution looks reasonable (more than 1 class).")

    except Exception as e:
        print(f"❌ Error reading file: {e}")


# 请修改为你实际的路径
train_path = './datasets/ninapro_db3_amputee/e3/train.pt'
val_path = './datasets/ninapro_db3_amputee/e3/val.pt'

inspect_dataset(train_path)
inspect_dataset(val_path)