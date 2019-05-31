import numpy as np
import os
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from preprocess import Preprocess
import math
from sortedcontainers import SortedList

class RecSysDataset(Dataset):
    def __init__(self, config, train_mode=True, toy_mode=True, valid_data=False):
        super(RecSysDataset, self).__init__()

        self.config = config
        self.train_mode = train_mode
        self.preprocessor = Preprocess(self.config, train_mode, toy_mode)
        self.item_vector_dim = len(self.preprocessor.meta_idx)
        self.item_vector_dict = self.preprocessor.item_vector_dict
        self.data_infos = self.preprocessor.data_infos
        self.session_keys = SortedList(self.data_infos[0].keys())
        if train_mode:
            if valid_data:
                self.session_keys = self.session_keys[int(config.experiment['data_ratio'] * len(self.session_keys)):]
            else:
                self.session_keys = self.session_keys[:int(config.experiment['data_ratio'] * len(self.session_keys))]

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
            try:
                instance['item_vectors'][j, :] = np.array(self.item_vector_dict[int(instance['impressions'][j])])
            except:
                ## todo : KeyError confirm
                instance['item_vectors'][j, :] = np.zeros((self.item_vector_dim))

        instance['session_vectors'] = self.data_infos[4][instance_key]
        instance['context_vectors'] = self.data_infos[5][instance_key]
        instance['prices'] = self.data_infos[6][instance_key]
        return instance

def _collate_fn(batch):
    batch_size = len(batch)
    keys = list()
    session_vector = list()
    time_infos = list()
    labels = list()
    item_idx = list()
    item_vectors = np.zeros((batch_size, 25, batch[0]['item_vectors'].shape[1]))

    lengths, sorted_idx = torch.tensor([len(batch[i]['session_vectors']) for i in range(batch_size)], dtype=torch.int64).sort(descending=True)
    max_len = lengths[0].item()
    session_vectors = np.zeros((batch_size, max_len, batch[0]['session_vectors'].shape[1]))
    session_context_vectors = np.zeros((batch_size, max_len, batch[0]['session_vectors'].shape[1]+batch[0]['context_vectors'].shape[1]))
    prices = np.zeros((batch_size, 25))

    idx = 0
    for i in sorted_idx:
        sample = batch[i]
        keys.append(sample['key'])
        session_vector.append(sample['session_vector'])
        time_infos.append(sample['time_info'])
        labels.append(sample['label'])
        item_idx.append(sample['impressions'])
        item_vectors[idx, :, :] = sample['item_vectors']

        session_vectors[idx, :lengths[idx], :] = sample['session_vectors']
        session_context_vectors[idx, :lengths[idx], :session_vectors.shape[-1]] = sample['session_vectors']
        session_context_vectors[idx, :lengths[idx], session_vectors.shape[-1]:] = sample['context_vectors']
        prices[idx,:len(sample['prices'])] = sample['prices']
        idx += 1

    session_vector = torch.tensor(session_vector, dtype=torch.float32)
    item_vectors = torch.tensor(item_vectors, dtype=torch.float32)

    session_vectors = pack_padded_sequence(torch.tensor(session_vectors, dtype=torch.float32), lengths, batch_first=True)
    session_context_vectors = pack_padded_sequence(torch.tensor(session_context_vectors, dtype=torch.float32), lengths, batch_first=True)
    prices = torch.tensor(prices, dtype=torch.float32)

    return keys, session_vector, time_infos, labels, item_idx, item_vectors, session_vectors, session_context_vectors, prices

class RecSysDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(RecSysDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

if __name__ == "__main__":
    from hparams import HParams
    config = HParams.load("hparams.yaml")
    print('train : ', config.mode['train'])
    dataset = RecSysDataset(config, train_mode=config.mode['train'], toy_mode=config.mode['toy'])
    print(len(dataset))

    dataloader = RecSysDataLoader(dataset, batch_size=512, num_workers=1, shuffle=False, drop_last=False)
    for batch in dataloader:
        keys = batch[0]

    print('end')
