import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from preprocess import Preprocess
import math
from sortedcontainers import SortedList

class RecSysDataset(Dataset):
    def __init__(self, config, train_mode=True, toy_mode=True):
        super(RecSysDataset, self).__init__()

        self.config = config
        self.train_mode = train_mode
        self.preprocessor = Preprocess(self.config, train_mode, toy_mode)
        self.item_vector_dim = len(self.preprocessor.meta_idx)
        self.item_vector_dict = self.preprocessor.item_vector_dict
        self.data_infos = self.preprocessor.data_infos
        self.session_keys = SortedList(self.data_infos[0].keys())

    def __len__(self):
        return len(self.session_keys)

    def __getitem__(self, idx):
        instance_key = self.session_keys[idx]
        instance = dict()

        instance['key'] = instance_key
        instance['session_vector'] = self.data_infos[0][instance_key]
        instance['time_info'] = self.data_infos[1][instance_key]
        if self.train_mode is True:
            instance['label'] = self.data_infos[3][instance_key]
        else:
            instance['label'] = -1
        instance['impressions'] = self.data_infos[2][instance_key]
        instance['item_vectors'] = np.zeros((25, self.item_vector_dim))
        for j in range(len(instance['impressions'])):
            instance['item_vectors'][j, :] = np.array(self.item_vector_dict[int(instance['impressions'][j])])

        return instance

def _collate_fn(batch):
    batch_size = len(batch)
    keys = list()
    session_vectors = list()
    time_infos = list()
    labels = list()
    item_idx = list()
    item_vectors = np.zeros((batch_size, 25, batch[0]['item_vectors'].shape[1]))

    for i in range(batch_size):
        sample = batch[i]
        keys.append(sample['key'])
        session_vectors.append(sample['session_vector'])
        time_infos.append(sample['time_info'])
        labels.append(sample['label'])
        item_idx.append(sample['impressions'])
        item_vectors[i, :, :] = sample['item_vectors']
    session_vectors = torch.tensor(session_vectors, dtype=torch.float32)
    item_vectors = torch.tensor(item_vectors, dtype=torch.float32)
    return keys, session_vectors, time_infos, labels, item_idx, item_vectors

class RecSysDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(RecSysDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

if __name__ == "__main__":
    from hparams import HParams
    config = HParams.load("hparams.yaml")
    dataset = RecSysDataset(config, train_mode=True, toy_mode=False)
    print(len(dataset))

    dataloader = RecSysDataLoader(dataset, batch_size=2, num_workers=1, shuffle=True, drop_last=False)
    for batch in dataloader:
        keys = batch[0]
        break
