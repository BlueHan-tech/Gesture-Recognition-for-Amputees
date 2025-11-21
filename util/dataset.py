from typing import Any, Tuple, Dict
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import util.augmentations as augmentations
import util.transformations as transformations


class TimeSeriesDataset(Dataset):
    """
    Dataset for multi-domain time series analysis.
    """

    def __init__(self, data_path: str, labels_path: str = None, labels_mask_path: str = None,
                 downstream_task: str = None,
                 domain_offsets: Dict = None, domain_agnostic: str = False,
                 univariate: bool = False,
                 train: bool = False,
                 test: bool = False,
                 N_val: int = 1,
                 args=None) -> None:

        data = torch.load(data_path, map_location=torch.device('cpu'))  # load to ram

        # ---------------------------
        # domain, data — 原逻辑保持不变
        # ---------------------------
        domain = [(sample[0], sample[1].unsqueeze(0).shape) for sample in data]
        data = [sample[1].unsqueeze(0) for sample in data]

        self.univariate = univariate
        self.domain_agnostic = True if self.univariate else domain_agnostic

        self.domain = domain
        self.domains = {domain: shape for domain, shape in sorted(list(set(self.domain)))}  # unique domains

        domain_list = [mod[0] for mod in domain]
        unique_domains = list(set(domain_list))

        self.domain_weights = {}
        for mod_current in unique_domains:
            mod_indices = torch.tensor([mod == mod_current for mod in domain_list])
            mod_weight = len(domain) / (len(unique_domains) * mod_indices.sum())
            self.domain_weights.update({mod_current: mod_weight})

        # offsets
        self.offsets = {}
        if domain_offsets is None:
            offset = 0
            for domain, shape in self.domains.items():
                self.offsets.update({domain: offset})
                if not self.domain_agnostic:
                    offset += shape[-2]
        else:
            self.offsets = domain_offsets

        self.data = data

        # ---------------------------------------------
        # 🔧 修复标签：优先使用 sample[2] 作为 label
        # ---------------------------------------------
        if len(data) > 0 and isinstance(data, list) and len(data_path) > 0:
            try:
                # 如果第三项存在 → 取 sample[2] 作为 label
                labels = torch.tensor([int(s[2]) for s in torch.load(data_path)], dtype=torch.long)
                self.labels = labels
                print("[dataset] ✔ Loaded labels from PT file (sample[2]).")
            except:
                # 若没有第三项则 fallback 到 labels_path（极少情况）
                if labels_path:
                    self.labels = torch.load(labels_path, map_location=torch.device('cpu'))
                    print("[dataset] ✔ Loaded labels from labels_path.")
                else:
                    self.labels = torch.zeros(size=(len(self.data),))
                    print("[dataset] ⚠ No labels found, defaulting to zeros.")
        else:
            self.labels = torch.zeros(size=(len(self.data),))

        # mask
        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu'))
        else:
            self.labels_mask = torch.ones_like(self.labels)

        self.downstream_task = downstream_task
        self.train = train
        self.test = test
        self.N_val = N_val
        self.args = args

    def set_domain_offsets(self, domain_offsets: Dict = None):
        self.offsets = domain_offsets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:

        data = self.data[idx]

        if self.univariate:
            data = data.permute(1, 0, 2)

        N = data.shape[0]

        if not self.train:
            stride = 32
            if self.test: stride = 1

            N_val = max(int(((data.shape[-1] - self.args.time_steps) / stride) + 1), 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            data_chunks = [data[..., i * stride:(i * stride + self.args.time_steps)] for i in range(N_val)]
            data = torch.cat(data_chunks, dim=0)

        else:
            N_val = max(data.shape[-1] - self.args.time_steps + 1, 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            transform = transforms.Compose([
                augmentations.CropResizing(fixed_resize_len=self.args.time_steps,
                                           lower_bnd=self.args.crop_lower_bnd,
                                           upper_bnd=self.args.crop_upper_bnd,
                                           resize=True),
                augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
                augmentations.Jitter(sigma=self.args.jitter_sigma),
                augmentations.Rescaling(sigma=self.args.rescaling_sigma),
            ])

            data_chunks = [transform(data) for i in range(N_val)]
            data = torch.cat(data_chunks, dim=0)

        if self.downstream_task == 'regression':
            label = self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd]
            label_mask = self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            label = self.labels[idx].type(torch.LongTensor)
            label_mask = torch.ones_like(label)

        domain, _ = self.domain[idx]
        domain_offset = self.offsets[domain]

        label = [label for i in range(N)]
        label_mask = [label_mask for i in range(N)]
        domain = [domain for i in range(N)]

        return data, label, label_mask, self.args.patch_size, domain_offset, domain, self.args.time_steps, self.univariate

    @staticmethod
    def collate_fn(batch):
        # unchanged …
        patch_size = batch[0][3]
        grid_width = torch.tensor([sample[0].shape[-1] // patch_size[-1] for sample in batch])
        grid_height = torch.tensor([sample[0].shape[-2] // patch_size[-2] for sample in batch])

        shape = [data.shape for sample in batch for data in sample[0]]
        max_values = [max(x) for x in zip(*shape)]
        max_variates = max_values[-2]
        max_timesteps = min(((max_values[-1] // patch_size[-1]) + 1) * patch_size[-1], batch[0][6])

        if grid_width.max() * patch_size[-1] < batch[0][6]:
            grid_width = grid_width + 1

        data = [torch.nn.functional.pad(
            data.unsqueeze(0),
            pad=(0, int(max_timesteps - data.shape[-1]), 0, int(max_variates - data.shape[-2])),
            mode="constant", value=0)
            for sample in batch for data in sample[0]]
        data = torch.stack(data, dim=0)

        attn_mask = [torch.nn.functional.pad(
            torch.ones(size=(grid_height[idx], grid_width[idx])),
            pad=(0, int(grid_width.max() - grid_width[idx]),
                 0, int(grid_height.max() - grid_height[idx])),
            mode="constant", value=0)
            for idx, sample in enumerate(batch) for data in sample[0]]
        attn_mask = torch.stack(attn_mask, dim=0)

        pos_embed_y = [torch.nn.functional.pad(
            torch.arange(grid_height[idx]).view(-1, 1).repeat(1, grid_width[idx]) + 1 + sample[4],
            pad=(0, int(grid_width.max() - grid_width[idx]),
                 0, int(grid_height.max() - grid_height[idx])),
            mode="constant", value=0)
            for idx, sample in enumerate(batch) for data in sample[0]]
        pos_embed_y = torch.stack(pos_embed_y, dim=0)

        domain = [domain for sample in batch for domain in sample[5]]

        return data, attn_mask, torch.LongTensor(pos_embed_y), domain

    @staticmethod
    def collate_fn_ft(batch):

        data = torch.stack([data.unsqueeze(0) for sample in batch for data in sample[0]], dim=0)
        label = torch.stack([label for sample in batch for label in sample[1]], dim=0)
        label_mask = torch.stack([label_mask for sample in batch for label_mask in sample[2]], dim=0)

        grid_width = batch[0][0].shape[-1] // batch[0][3][-1]
        grid_height = batch[0][0].shape[-2] // batch[0][3][-2]

        pos_embed_y = torch.arange(grid_height).view(-1, 1).repeat(len(batch), 1, grid_width) + 1 + batch[0][4]
        pos_embed_y = torch.stack([pos_embed_y[idx] for idx, sample in enumerate(batch) for data in sample[0]], dim=0)

        return data, label, label_mask, torch.LongTensor(pos_embed_y)
