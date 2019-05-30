import math
import pandas as pd
import numpy as np
import os
from sortedcontainers import SortedList
import torch

GR_COLS = ["user_id", "session_id", "timestamp", "step"]
action_types = SortedList(['clickout item', 'interaction item rating', 'interaction item info', 'interaction item image', 'interaction item deals', 'change of sort order', 'filter selection', 'search for item', 'search for destination', 'search for poi'])
action_idx = dict()
for i in range(len(action_types)):
    action_idx[action_types[i]] = i

class Preprocess():
    def __init__(self, config, train_mode, toy_mode):
        self.config = config

        if train_mode:
            if toy_mode:
                data_name = 'toy_train'
            else:
                data_name = 'train'
        else:
            if toy_mode:
                data_name = 'toy_test'
            else:
                data_name = 'test'

        meta_path = os.path.join(self.config.root_path, 'data', 'meta.pt')
        if os.path.exists(meta_path):
            meta = torch.load(meta_path)
            self.meta_idx = meta['idx']
            self.item_vector_dict = meta['item_vector']
        else:
            self.meta_data_df = pd.read_csv(os.path.join(self.config.root_path, 'data', 'item_metadata.csv'))
            self.meta_idx = self.make_meta_idx()
            self.item_vector_dict = self.make_item_vector_dict()
            meta = dict()
            meta['idx'] = self.meta_idx
            meta['item_vector'] = self.item_vector_dict
            torch.save(meta,meta_path)

        preprocess_save_path = os.path.join(self.config.root_path, 'data', data_name + '.pt')
        if os.path.exists(preprocess_save_path):
            self.data_infos = torch.load(preprocess_save_path)
        else:
            self.data_df = pd.read_csv(os.path.join(self.config.root_path, 'data', data_name + '.csv'))
            self.data_infos = self.preprocess_sessions(self.data_df, train=train_mode)
            torch.save(self.data_infos, preprocess_save_path)

    def make_meta_idx(self):
        attributes = set()
        for i in range(len(self.meta_data_df)):
            tmp_list = self.meta_data_df.iloc[i,1].split('|')
            attributes.update(tmp_list)
        attributes = SortedList(attributes)
        meta_idx = dict()
        for i in range(len(attributes)):
            meta_idx[attributes[i]] = i
        return meta_idx

    def make_item_vector_dict(self):
        vector_dim = len(self.meta_idx)
        item_vector_dict = dict()
        for i in range(len(self.meta_data_df)):
            tmp_list = self.meta_data_df.iloc[i,1].split('|')
            tmp_array = np.zeros(vector_dim)
            for t in tmp_list:
                tmp_array[self.meta_idx[t]] = 1
            item_vector_dict[int(self.meta_data_df.iloc[i,0])] = tmp_array
        return item_vector_dict

    def preprocess_sessions(self, data_df, train=True):
        session_vector_dict = dict()
        session_time_dict = dict()
        session_impressions_dict = dict()
        session_label_dict = dict()

        session_vectors_dict = dict()
        session_contexts_dict = dict()
        impressions_price_dict = dict()
        num_sort_order = 3

        i = 0
        while i < len(data_df):
            tmp_list = list(data_df.iloc[i])
            tmp_user, tmp_session = tmp_list[0], tmp_list[1]
            user, session = tmp_list[0], tmp_list[1]
            tmp_array = np.zeros(len(action_idx))

            tmp_vectors_list = list()
            tmp_context_list = list()

            while tmp_user == user and tmp_session == session:
                tmp_list = list(data_df.iloc[i])
                tmp_array[action_idx[tmp_list[4]]] += 1

                tmp = np.zeros(len(action_idx))
                tmp[action_idx[tmp_list[4]]] += 1

                tmp_context = np.zeros(len(self.meta_idx) + num_sort_order)

                if action_idx[tmp_list[4]] == 0:
                    if tmp_list[5].find('rating') != -1:
                        tmp_context[-3] = 1
                    elif tmp_list[5].find('price') != -1:
                        tmp_context[-2] = 1
                    elif tmp_list[5].find('distance') != -1:
                        tmp_context[-1] = 1
                elif action_idx[tmp_list[4]] >= 3 and action_idx[tmp_list[4]] <=6:
                    try:
                        tmp_context[:len(self.meta_idx)] = self.item_vector_dict[tmp_list[5]]
                    except:
                        tmp_context[:len(self.meta_idx)] = np.zeros(len(self.meta_idx))
                elif action_idx[tmp_list[4]] == 8:
                    try:
                        tmp_context[:len(self.meta_idx)] = self.item_vector_dict[tmp_list[5]]
                    except:
                        tmp_context[:len(self.meta_idx)] = np.zeros(len(self.meta_idx))

                tmp_vectors_list.append(tmp)
                tmp_context_list.append(tmp_context)

                i += 1
                if i < len(data_df):
                    user, session = list(data_df.iloc[i])[0], list(data_df.iloc[i])[1]
                else: break
                if list(data_df.iloc[i])[3] == 1:
                    break

            if tmp_list[4] == 'clickout item':
                if train is True:
                    session_vector_dict[tmp_user, tmp_session] = tmp_array
                    session_time_dict[tmp_user, tmp_session] = tmp_list[2:4]
                    session_impressions_dict[tmp_user, tmp_session] = tmp_list[10].split('|')
                    session_label_dict[tmp_user, tmp_session] = int(tmp_list[5])

                    session_vectors_dict[tmp_user, tmp_session] = np.array(tmp_vectors_list)
                    session_contexts_dict[tmp_user, tmp_session] = np.array(tmp_context_list)
                    impressions_price_dict[tmp_user, tmp_session] = np.array(tmp_list[11].split('|'), dtype=int)
                else:
                    if np.isnan(float(tmp_list[5])):
                        session_vector_dict[tmp_user, tmp_session] = tmp_array
                        session_time_dict[tmp_user, tmp_session] = tmp_list[2:4]
                        session_impressions_dict[tmp_user, tmp_session] = tmp_list[10].split('|')

                        session_vectors_dict[tmp_user, tmp_session] = tmp_vectors_list
                        session_contexts_dict[tmp_user, tmp_session] = np.array(tmp_context_list)
                        impressions_price_dict[tmp_user, tmp_session] = np.array(tmp_list[11].split('|'), dtype=int)
        return session_vector_dict, session_time_dict, session_impressions_dict, session_label_dict, session_vectors_dict, session_contexts_dict, impressions_price_dict

if __name__ == "__main__":
    from hparams import HParams
    config = HParams.load("hparams.yaml")
    print('train : ', config.mode['train'])
    proc = Preprocess(config,train_mode=config.mode['train'], toy_mode=config.mode['toy'])
    print('end')













