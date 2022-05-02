"""
    PartNetPartDataset
"""

import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
import ipdb
from pathlib import Path
from typing import List, Optional

from src.utils.paths import DATA_FOLDER

COMMON_DATA_FEATURES = ('part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'pairs')


class PartNetDataset(data.Dataset):

    def __init__(
            self,
            category: str,
            data_file: Path,
            level: str = '3',
            data_features: List[str] = COMMON_DATA_FEATURES,
            max_num_part: int = 20,
            filter_out_bad_examples: bool = True
    ):

        # store parameters
        self.category = category

        self.max_num_part = max_num_part
        self.max_pairs = max_num_part * (max_num_part - 1) / 2
        self.level = level

        # load data
        self.data_file = data_file
        self.data = np.load(str(data_file))

        # data features
        self.data_features = data_features

        # load category semantics information
        self.part_sems = []
        self.part_sem2id = dict()

        if filter_out_bad_examples:
            self.clean_examples()

    def clean_examples(self):
        good_indices = []
        for idx, shape_id in enumerate(self.data):
            if self.get_part_pcs(shape_id) is not None:
                good_indices.append(idx)

        self.data = np.take(self.data, good_indices, axis=0)

    def get_part_count(self) -> int:
        return len(self.part_sems)

    def load_contact_data(self, shape_id: int) -> np.array:
        cur_contact_data_fn = DATA_FOLDER / \
                              f'contact_points/pairs_with_contact_points_{str(shape_id)}_level{self.level}.npy'
        cur_contacts = np.load(cur_contact_data_fn, allow_pickle=True)
        return cur_contacts

    def load_shape_data(self, shape_id: int) -> np.array:
        cur_data_fn = DATA_FOLDER / f'shape_data/{str(shape_id)}_level{self.level}.npy'
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()  # assume data is stored in seperate .npz file
        return cur_data

    def get_contact_points(self, shape_id: int, contact_data: Optional[np.array] = None) -> torch.Tensor:
        if not contact_data:
            contact_data = self.load_contact_data(shape_id)

        cur_num_part = contact_data.shape[0]
        out = np.zeros((self.max_num_part, self.max_num_part, 4), dtype=np.float32)
        out[:cur_num_part, :cur_num_part, :] = contact_data
        out = torch.from_numpy(out).float().unsqueeze(0)

        return out

    def get_syms(self, shape_id: int, shape_data: Optional[np.array] = None) -> Optional[torch.Tensor]:
        if not shape_data:
            shape_data = self.load_shape_data(shape_id)

        cur_sym = shape_data['sym']
        cur_num_part = cur_sym.shape[0]

        # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data
        # item
        if cur_num_part > self.max_num_part:
            return None

        out = np.zeros((self.max_num_part, cur_sym.shape[1]), dtype=np.float32)
        out[:cur_num_part] = cur_sym
        out = torch.from_numpy(out).float().unsqueeze(0)  # p x 3

        return out

    def get_semantic_ids(self, shape_id, shape_data: Optional[np.array] = None) -> Optional[torch.Tensor]:
        if not shape_data:
            shape_data = self.load_shape_data(shape_id)

        cur_part_ids = shape_data['part_ids']
        cur_num_part = len(cur_part_ids)

        # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data
        # item
        if cur_num_part > self.max_num_part:
            return None

        out = np.zeros((self.max_num_part), dtype=np.float32)
        out[:cur_num_part] = cur_part_ids
        out = torch.from_numpy(out).float().unsqueeze(0)  # 1 x 20
        return out

    def get_part_pcs(self, shape_id: int, shape_data: Optional[np.array] = None) -> Optional[torch.Tensor]:
        if not shape_data:
            shape_data = self.load_shape_data(shape_id)

        cur_pts = shape_data['part_pcs']  # p x N x 3 (p is unknown number of parts for this shape)
        cur_num_part = cur_pts.shape[0]

        # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data
        # item
        if cur_num_part > self.max_num_part:
            return None

        out = np.zeros((self.max_num_part, cur_pts.shape[1], 3), dtype=np.float32)
        out[:cur_num_part] = cur_pts
        out = torch.from_numpy(out).float().unsqueeze(0)  # 1 x 20 x N x 3

        return out

    def get_part_poses(self, shape_id: int, shape_data: Optional[np.array] = None) -> Optional[torch.Tensor]:
        if not shape_data:
            shape_data = self.load_shape_data(shape_id)

        cur_pose = shape_data['part_poses']  # p x (3 + 4)
        cur_num_part = cur_pose.shape[0]

        # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data
        # item
        if cur_num_part > self.max_num_part:
            return None

        out = np.zeros((self.max_num_part, 3 + 4), dtype=np.float32)
        out[:cur_num_part] = cur_pose
        out = torch.from_numpy(out).float().unsqueeze(0)  # 1 x 20 x (3 + 4)

        return out

    def get_part_valids(self, shape_id: int, shape_data: Optional[np.array] = None) -> Optional[torch.Tensor]:
        if not shape_data:
            shape_data = self.load_shape_data(shape_id)

        cur_pose = shape_data['part_poses']  # p x (3 + 4)
        cur_num_part = cur_pose.shape[0]

        # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data
        # item
        if cur_num_part > self.max_num_part:
            return None

        out = np.zeros((self.max_num_part), dtype=np.float32)
        out[:cur_num_part] = 1
        out = torch.from_numpy(out).float().unsqueeze(0)  # 1 x 20 (return 1 for the first p parts, 0 for the rest)

        return out

    def get_part_ids(self, shape_id: int, shape_data: Optional[np.array] = None) -> Optional[torch.Tensor]:
        if not shape_data:
            shape_data = self.load_shape_data(shape_id)

        cur_part_ids = shape_data['geo_part_ids']
        cur_pose = shape_data['part_poses']  # p x (3 + 4)
        cur_num_part = cur_pose.shape[0]

        # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data
        # item
        if cur_num_part > self.max_num_part:
            return None

        out = np.zeros((self.max_num_part), dtype=np.float32)
        out[:cur_num_part] = cur_part_ids
        out = torch.from_numpy(out).float().unsqueeze(0)  # 1 x 20

        return out

    def get_pairs(self, shape_id: int, shape_data: Optional[np.array] = None) -> Optional[torch.Tensor]:
        if not shape_data:
            shape_data = self.load_shape_data(shape_id)

        cur_pose = shape_data['part_poses']  # p x (3 + 4)
        cur_num_part = cur_pose.shape[0]

        # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data
        # item
        if cur_num_part > self.max_num_part:
            return None

        cur_pose = shape_data['part_poses']  # p x (3 + 4)
        cur_vaild_num = len(cur_pose)
        valid_pair_martix = np.ones((cur_vaild_num, cur_vaild_num))
        pair_martix = np.zeros((self.max_num_part, self.max_num_part))
        pair_martix[:cur_vaild_num, :cur_vaild_num] = valid_pair_martix
        out = torch.from_numpy(pair_martix).unsqueeze(0)

        return out

    def get_match_ids(self, shape_id: int, shape_data: Optional[np.array] = None) -> Optional[np.array]:
        if not shape_data:
            shape_data = self.load_shape_data(shape_id)

        cur_part_ids = shape_data['geo_part_ids']
        cur_pose = shape_data['part_poses']  # p x (3 + 4)
        cur_num_part = cur_pose.shape[0]

        # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data
        # item
        if cur_num_part > self.max_num_part:
            return None

        out = np.zeros((self.max_num_part), dtype=np.float32)
        out[:cur_num_part] = cur_part_ids
        index = 1
        for i in range(1, 58):
            idx = np.where(out == i)[0]
            idx = torch.from_numpy(idx)
            # print(idx)
            if len(idx) == 0:
                continue
            elif len(idx) == 1:
                out[idx] = 0
            else:
                out[idx] = index
                index += 1

        return out

    def __str__(self):
        strout = '[PartNetPartDataset %s %d] data_file: %s, max_num_part: %d' % \
                 (self.category, len(self), self.data_file, self.max_num_part)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        shape_id = self.data[index]
        shape_data = self.load_shape_data(shape_id)

        data_feats = ()

        # This is a loop due to the training file depending on the feature data being at the same index as
        # specified in the data_features argument (i.e. sym must be at the same index in the output of this
        # function as it was specified in self.data_features)
        for feature in self.data_features:
            if feature == 'contact_points':
                contact_points = self.get_contact_points(shape_id)
                data_feats = data_feats + (contact_points,)

            elif feature == 'sym':
                syms = self.get_syms(shape_id, shape_data=shape_data)
                data_feats = data_feats + (syms,)

            elif feature == 'semantic_ids':
                semantic_ids = self.get_semantic_ids(shape_id, shape_data=shape_data)
                data_feats = data_feats + (semantic_ids,)

            elif feature == 'part_pcs':
                part_pcs = self.get_part_pcs(shape_id, shape_data=shape_data)
                data_feats = data_feats + (part_pcs,)

            elif feature == 'part_poses':
                part_poses = self.get_part_poses(shape_id, shape_data=shape_data)
                data_feats = data_feats + (part_poses,)

            elif feature == 'part_valids':
                part_valids = self.get_part_valids(shape_id, shape_data=shape_data)
                data_feats = data_feats + (part_valids,)

            elif feature == 'shape_id':
                data_feats = data_feats + (shape_id,)

            elif feature == 'part_ids':
                part_ids = self.get_part_ids(shape_id, shape_data=shape_data)
                data_feats = data_feats + (part_ids,)

            elif feature == 'pairs':
                pairs = self.get_pairs(shape_id, shape_data=shape_data)
                data_feats = data_feats + (pairs,)

            elif feature == 'match_ids':
                match_ids = self.get_match_ids(shape_id, shape_data=shape_data)
                data_feats = data_feats + (match_ids,)

        return data_feats


if __name__ == "__main__":
    dataset = PartNetDataset(
        category='',
        data_file=DATA_FOLDER / 'Chair.train.npy',
        max_num_part=20,
    )

    print(dataset[0])