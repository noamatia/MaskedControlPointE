import os
import tqdm
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from point_e.util.point_cloud import PointCloud

PROMPTS = "prompt"
SOURCE_UID = "source_uid"
TARGET_UID = "target_uid"
SOURCE_MASKS = "source_masks"
TARGET_MASKS = "target_masks"
SOURCE_LATENTS = "source_latents"
TARGET_LATENTS = "target_latents"
PARTNET_DIR = "/scratch/noam/data_v0"
PARTNET_MASKED_LABELS_DIR = "data/partnet"
PCS_DIR = "/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering"


def masked_labels_path(uid):
    return os.path.join(PARTNET_MASKED_LABELS_DIR, uid, "masked_labels.json")


def load_partnet_metadata(uid, part, masked):
    path = masked_labels_path(uid)
    with open(path, "r") as f:
        data = json.load(f)
    masked_labels = data["masked_labels"][part] if masked else None
    return masked_labels, data["partnet_uid"]


def load_masked_pc(partnet_uid, num_points, masked_labels):
    src_dir = os.path.join(PARTNET_DIR, partnet_uid)
    pc = PointCloud.load_masked(
        os.path.join(src_dir, "point_sample", "pts-10000.txt"),
        labels_path=os.path.join(src_dir, "point_sample", "label-10000.txt"),
        masked_labels=masked_labels,
    )
    return pc.random_sample(num_points)


class MaskedControlShapeNet(Dataset):
    def __init__(
        self,
        part: str,
        masked: bool,
        num_points: int,
        batch_size: int,
        df: pd.DataFrame,
        prompt_key: str,
        device: torch.device,
    ):
        super().__init__()
        self.prompts = []
        self.source_uids = []
        self.target_uids = []
        self.source_masks = []
        self.target_masks = []
        self.source_latents = []
        self.target_latents = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating data"):
            self._append_sample(row, prompt_key, num_points, device, part, masked)
        self._set_length(batch_size)

    def _append_sample(self, row, prompt_key, num_points, device, part, masked):
        prompt, source_uid, target_uid = (
            row[prompt_key],
            row[SOURCE_UID],
            row[TARGET_UID],
        )
        source_masked_labels, source_partnet_uid = load_partnet_metadata(
            source_uid, part, masked
        )
        target_masked_labels, target_partnet_uid = load_partnet_metadata(
            target_uid, part, masked
        )
        self.prompts.append(prompt)
        self.source_uids.append(source_uid)
        self.target_uids.append(target_uid)
        source_pc = load_masked_pc(
            source_partnet_uid,
            num_points,
            source_masked_labels,
        )
        target_pc = load_masked_pc(
            target_partnet_uid,
            num_points,
            target_masked_labels,
        )
        self.source_latents.append(source_pc.encode().to(device))
        self.target_latents.append(target_pc.encode().to(device))
        if masked:
            self.source_masks.append(source_pc.encode_mask().to(device))
            self.target_masks.append(target_pc.encode_mask().to(device))
        else:
            self.source_masks.append(0)
            self.target_masks.append(0)

    def _set_length(self, batch_size):
        self.length = len(self.prompts)
        r = self.length % batch_size
        if r == 0:
            self.logical_length = self.length
        else:
            q = batch_size - r
            self.logical_length = self.length + q

    def __len__(self):
        return self.logical_length

    def __getitem__(self, logical_index):
        index = logical_index % self.length
        return {
            PROMPTS: self.prompts[index],
            SOURCE_MASKS: self.source_masks[index],
            TARGET_MASKS: self.target_masks[index],
            SOURCE_LATENTS: self.source_latents[index],
            TARGET_LATENTS: self.target_latents[index],
        }
