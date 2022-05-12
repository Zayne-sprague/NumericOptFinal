from typing import List, Optional
import torch.utils.data as data
import torch
import numpy as np
from random import choice

from src.datasets.partnet_dataset import PartNetDataset, COMMON_DATA_FEATURES

COMMON_DISTRACTOR_DATA_FEATURES = ('part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'pairs', 'part_labels')

class PartNetDistractorDataset(data.Dataset):

    def __init__(
            self,
            training_dataset: PartNetDataset,
            distractor_dataset: Optional[PartNetDataset] = None,
            data_features: List[str] = COMMON_DISTRACTOR_DATA_FEATURES,
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
        strout = 'Distractor Dataset'
        return strout

    def __len__(self):
        return len(self.training_dataset)

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

        random_indices = list(range(self.training_dataset.max_num_part + self.distractor_dataset.max_num_part))
        np.random.shuffle(random_indices)

        for feature in self.data_features:
            if feature == 'contact_points':
                gold_contact_points = self.training_dataset.get_contact_points(gold_shape_id)
                distractor_contact_points = self.distractor_dataset.get_contact_points(distractor_shape_id)

                if gold_contact_points is None or distractor_contact_points is None:
                    data_feats += (None,)
                else:
                    contact_points = torch.cat([gold_contact_points, distractor_contact_points], dim=1)
                    contact_points[:, :, :] = contact_points[:, random_indices, :]
                    data_feats = data_feats + (contact_points,)

            elif feature == 'sym':
                gold_syms = self.training_dataset.get_syms(gold_shape_id, shape_data=gold_shape_data)
                distractor_syms = self.distractor_dataset.get_syms(ditractor_shape_id, shape_data=distractor_shape_data)

                if gold_syms is None or distractor_syms is None:
                    data_feats += (None,)
                else:
                    syms = torch.cat([gold_syms, distractor_syms], dim=1)
                    syms[:, :, :] = syms[:, random_indices, :]
                    data_feats = data_feats + (syms,)

            elif feature == 'semantic_ids':
                gold_semantic_ids = self.training_dataset.get_semantic_ids(gold_shape_id, shape_data=gold_shape_data)
                distractor_semantic_ids = self.distractor_dataset.get_semantic_ids(
                    distractor_shape_id,
                    shape_data=distractor_shape_data
                )

                if gold_semantic_ids is None or distractor_semantic_ids is None:
                    data_feats += (None,)
                else:
                    semantic_ids = torch.cat([gold_semantic_ids, distractor_semantic_ids], dim=1)
                    semantic_ids[:, :, :] = semantic_ids[:, random_indices, :]
                    data_feats = data_feats + (semantic_ids,)

            elif feature == 'part_pcs':
                gold_part_pcs = self.training_dataset.get_part_pcs(gold_shape_id, shape_data=gold_shape_data)
                distractor_part_pcs = self.distractor_dataset.get_part_pcs(
                    distractor_shape_id,
                    shape_data=distractor_shape_data
                )

                if gold_part_pcs is None or distractor_part_pcs is None:
                    data_feats += (None,)
                else:
                    part_pcs = torch.cat([gold_part_pcs, distractor_part_pcs], dim=1)
                    part_pcs[:, :, :, :] = part_pcs[:, random_indices, :, :]
                    data_feats = data_feats + (part_pcs,)

            elif feature == 'part_poses':
                gold_part_poses = self.training_dataset.get_part_poses(gold_shape_id, shape_data=gold_shape_data)
                distractor_part_poses = self.distractor_dataset.get_part_poses(
                    distractor_shape_id,
                    shape_data=distractor_shape_data
                )

                if gold_part_poses is None or distractor_part_poses is None:
                    data_feats += (None,)
                else:
                    part_poses = torch.cat([gold_part_poses, distractor_part_poses], dim=1)
                    part_poses[:, :, :] = part_poses[:, random_indices, :]
                    data_feats = data_feats + (part_poses,)

            elif feature == 'part_valids':
                gold_part_valids = self.training_dataset.get_part_valids(gold_shape_id, shape_data=gold_shape_data)
                distractor_part_valids = self.distractor_dataset.get_part_valids(
                    distractor_shape_id,
                    shape_data=distractor_shape_data
                )

                if gold_part_valids is None or distractor_part_valids is None:
                    data_feats += (None,)
                else:
                    part_valids = torch.cat([gold_part_valids, distractor_part_valids], dim=1)
                    part_valids[:, :] = part_valids[:, random_indices]
                    data_feats = data_feats + (part_valids,)

            elif feature == 'shape_id':
                data_feats = data_feats + ((gold_shape_id, distractor_shape_id),)

            elif feature == 'part_ids':
                gold_part_ids = self.training_dataset.get_part_ids(gold_shape_id, shape_data=gold_shape_data)
                distractor_part_ids = self.distractor_dataset.get_part_ids(
                    distractor_shape_id,
                    shape_data=distractor_shape_data
                )

                if gold_part_ids is None or distractor_part_ids is None:
                    data_feats += (None,)
                else:
                    part_ids = torch.cat([gold_part_ids, distractor_part_ids], dim=1)
                    part_ids[:, :] = part_ids[:, random_indices]
                    data_feats = data_feats + (part_ids,)

            elif feature == 'pairs':
                gold_pairs = self.training_dataset.get_pairs(gold_shape_id, shape_data=gold_shape_data)
                distractor_pairs = self.distractor_dataset.get_pairs(distractor_shape_id, shape_data=distractor_shape_data)

                if gold_pairs is None or distractor_pairs is None:
                    data_feats += (None,)
                else:
                    pairs = torch.zeros([
                        1,
                        gold_pairs.shape[1] + distractor_pairs.shape[1],
                        gold_pairs.shape[2] + distractor_pairs.shape[2]
                    ]).double()
                    pairs[0:gold_pairs.shape[0], 0:gold_pairs.shape[1], 0:gold_pairs.shape[2]] += gold_pairs
                    pairs[:, gold_pairs.shape[1]:, gold_pairs.shape[2]:] += distractor_pairs

                    pairs[:, :, :] = pairs[:, :, random_indices]
                    pairs[:, :, :] = pairs[:, random_indices, :]

                    data_feats = data_feats + (pairs,)

            elif feature == 'match_ids':
                gold_match_ids = self.training_dataset.get_match_ids(gold_shape_id, shape_data=gold_shape_data)
                distractor_match_ids = self.distractor_dataset.get_match_ids(
                    distractor_shape_id,
                    shape_data=distractor_shape_data
                )

                if gold_match_ids is None or distractor_match_ids is None:
                    data_feats += (None,)
                else:
                    match_ids = np.concatenate([gold_match_ids, distractor_match_ids])
                    match_ids[:] = match_ids[random_indices]
                    data_feats = data_feats + (match_ids,)
            elif feature == 'part_labels':
                gold_labels = [0] * self.training_dataset.max_num_part
                distractor_labels = [1] * self.distractor_dataset.max_num_part

                labels = torch.tensor([*gold_labels, *distractor_labels])
                labels[:] = labels[random_indices]
                data_feats = data_feats + (labels,)

        return data_feats


if __name__ == "__main__":
    from src.utils.paths import DATA_FOLDER

    training_dataset = PartNetDataset(category='', data_file=DATA_FOLDER / 'Chair.test.npy', max_size=4)
    distractor_dataset = PartNetDataset(category='', data_file=DATA_FOLDER / 'Lamp.test.npy', max_num_part=1, max_size = 4)

    dataset = PartNetDistractorDataset(
        training_dataset=training_dataset,
        distractor_dataset=distractor_dataset,
    )

    for data in dataset:
        print("HI")