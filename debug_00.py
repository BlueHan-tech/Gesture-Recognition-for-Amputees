# debug_00.py
import torch

paths = [
    './datasets/ninapro_db3_amputee/e1/train.pt',
    './datasets/ninapro_db3_amputee/e1/val.pt',
    './datasets/ninapro_db3_amputee/e2/train.pt',
    './datasets/ninapro_db3_amputee/e2/val.pt',
]

for p in paths:
    print(f"\n=== 检查文件: {p} ===")
    data = torch.load(p, map_location='cpu', weights_only=False)  # 或者加 weights_only=True 也行

    # 情况1：直接是一个 list（极少见）
    if isinstance(data, list):
        labels = torch.tensor([item[1] for item in data])  # 根据 WaveFormer dataset 返回的 tuple 调整索引

    # 情况2：大多数人保存的 dict 格式（99%概率）
    elif isinstance(data, dict):
        if 'labels' in data:
            labels = data['labels']
        elif 'targets' in data:
            labels = data['targets']
        else:
            print("dict 中没找到 labels/targets 字段！")
            print("keys:", data.keys())
            continue
        # data['data'] 或者 data['samples'] 是信号
        signals = data.get('data') or data.get('samples') or data.get('signal')
        print(f"信号 shape: {signals.shape if signals is not None else 'None'}")

    else:
        print("未知数据格式！")
        print(type(data))
        continue

    # 现在 labels 一定是 tensor 了
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    unique_labels = torch.unique(labels)
    num_samples = len(labels)
    print(f"样本总数: {num_samples}")
    print(f"唯一类别数: {len(unique_labels)}")
    print(f"类别列表: {unique_labels.tolist()}")
    print(f"各类别样本数:")
    for cls in unique_labels:
        count = (labels == cls).sum().item()
        print(f"  类 {cls.item():2d}: {count:5d} 个样本")