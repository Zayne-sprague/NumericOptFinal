from typing import List, Optional
import torch.utils.data as data
import torch
import numpy as np
from random import choice

from src.datasets.partnet_dataset import PartNetDataset, COMMON_DATA_FEATURES


class PartNetDistractorDataset(data.Dataset):

    def __init__(
            self,
            training_dataset: PartNetDataset,
            distractor_dataset: Optional[PartNetDataset] = None,
            data_features: List[str] = COMMON_DATA_FEATURES,
    ):
        # This dataset will be used in the getitem call to pull the gold parts from
        self.training_dataset = training_dataset

        # This dataset will be used for distractor parts.  If a distractor dataset isn't supplied, it will use the
        # training dataset
        self.distractor_dataset = distractor_dataset if distractor_dataset is not None else self.training_dataset

        # Important to know so we can make sure to generate a different random index in the getitem call
        # (we don't want them to overlap)
        self.using_training_as_distractor = self.training_dataset.data_file == self.distractor_dataset.data_file

        self.data_features = data_features

    def __str__(self):
        strout = '[PartNetPartDataset %s %d] data_file: %s, max_num_part: %d' % \
                 (self.category, len(self), self.data_file, self.max_num_part)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        distractor_data_indices = set(range(len(self.distractor_dataset)))
        if self.using_training_as_distractor:
            distractor_data_indices.remove(index)
        distractor_idx = choice(list(distractor_data_indices))

        gold_shape_id = self.training_dataset.data[index]
        distractor_shape_id = self.distractor_dataset.data[distractor_idx]

        data = self.pair_gold_and_distractor(gold_shape_id, distractor_shape_id)

        return data

    def pair_gold_and_distractor(self, gold_shape_id: int, distractor_shape_id: int):
        gold_shape_data = self.training_dataset.load_shape_data(gold_shape_id)
        distractor_shape_data = self.distractor_dataset.load_shape_data(distractor_shape_id)

        data_feats = ()

        if 'contact_points' in self.data_features:
            gold_contact_points = self.training_dataset.get_contact_points(gold_shape_id)
            distractor_contact_points = self.distractor_dataset.get_contact_points(distractor_shape_id)

            if gold_contact_points is None or distractor_contact_points is None:
                data_feats += (None,)
            else:
                data_feats = data_feats + (torch.cat([gold_contact_points, distractor_contact_points], dim=1),)

        if 'sym' in self.data_features:
            gold_syms = self.training_dataset.get_syms(gold_shape_id, shape_data=gold_shape_data)
            distractor_syms = self.distractor_dataset.get_syms(ditractor_shape_id, shape_data=distractor_shape_data)

            if gold_syms is None or distractor_syms is None:
                data_feats += (None,)
            else:
                data_feats = data_feats + (torch.cat([gold_syms, distractor_syms], dim=1),)

        if 'semantic_ids' in self.data_features:
            gold_semantic_ids = self.training_dataset.get_semantic_ids(gold_shape_id, shape_data=gold_shape_data)
            distractor_semantic_ids = self.distractor_dataset.get_semantic_ids(
                distractor_shape_id,
                shape_data=distractor_shape_data
            )

            if gold_semantic_ids is None or distractor_semantic_ids is None:
                data_feats += (None,)
            else:
                data_feats = data_feats + (torch.cat([gold_semantic_ids, distractor_semantic_ids], dim=1),)

        if 'part_pcs' in self.data_features:
            gold_part_pcs = self.training_dataset.get_part_pcs(gold_shape_id, shape_data=gold_shape_data)
            distractor_part_pcs = self.distractor_dataset.get_part_pcs(
                distractor_shape_id,
                shape_data=distractor_shape_data
            )

            if gold_part_pcs is None or distractor_part_pcs is None:
                data_feats += (None,)
            else:
                data_feats = data_feats + (torch.cat([gold_part_pcs, distractor_part_pcs], dim=1),)

        if 'part_poses' in self.data_features:
            gold_part_poses = self.training_dataset.get_part_poses(gold_shape_id, shape_data=gold_shape_data)
            distractor_part_poses = self.distractor_dataset.get_part_poses(
                distractor_shape_id,
                shape_data=distractor_shape_data
            )

            if gold_part_poses is None or distractor_part_poses is None:
                data_feats += (None,)
            else:
                data_feats = data_feats + (torch.cat([gold_part_poses, distractor_part_poses], dim=1),)

        if 'part_valids' in self.data_features:
            gold_part_valids = self.training_dataset.get_part_valids(gold_shape_id, shape_data=gold_shape_data)
            distractor_part_valids = self.distractor_dataset.get_part_valids(
                distractor_shape_id,
                shape_data=distractor_shape_data
            )

            if gold_part_valids is None or distractor_part_valids is None:
                data_feats += (None,)
            else:
                data_feats = data_feats + (torch.cat([gold_part_valids, distractor_part_valids], dim=1),)

        if 'shape_id' in self.data_features:
            data_feats = data_feats + ((gold_shape_id, distractor_shape_id),)

        if 'part_ids' in self.data_features:
            gold_part_ids = self.training_dataset.get_part_ids(gold_shape_id, shape_data=gold_shape_data)
            distractor_part_ids = self.distractor_dataset.get_part_ids(
                distractor_shape_id,
                shape_data=distractor_shape_data
            )

            if gold_part_ids is None or distractor_part_ids is None:
                data_feats += (None,)
            else:
                data_feats = data_feats + (torch.cat([gold_part_ids, distractor_part_ids], dim=1),)

        if 'pairs' in self.data_features:
            gold_pairs = self.training_dataset.get_pairs(gold_shape_id, shape_data=gold_shape_data)
            distractor_pairs = self.distractor_dataset.get_pairs(distractor_shape_id, shape_data=distractor_shape_data)
            data_feats = data_feats + ((gold_pairs, distractor_pairs),)

        if 'match_ids' in self.data_features:
            gold_match_ids = self.training_dataset.get_match_ids(gold_shape_id, shape_data=gold_shape_data)
            distractor_match_ids = self.distractor_dataset.get_match_ids(
                distractor_shape_id,
                shape_data=distractor_shape_data
            )

            if gold_match_ids is None or distractor_match_ids is None:
                data_feats += (None,)
            else:
                data_feats = data_feats + (np.concatenate([gold_match_ids, distractor_match_ids]),)

        return data_feats


if __name__ == "__main__":
    from src.utils.paths import DATA_FOLDER

    training_dataset = PartNetDataset(category='', data_file=DATA_FOLDER / 'Chair.test.npy')
    distractor_dataset = PartNetDataset(category='', data_file=DATA_FOLDER / 'Lamp.test.npy', max_num_part=1)

    dataset = PartNetDistractorDataset(
        training_dataset=training_dataset,
        distractor_dataset=distractor_dataset,
    )

    for data in dataset:
        print("HI")