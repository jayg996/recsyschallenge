from utils.hparams import HParams
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import time

use_cuda = torch.cuda.is_available()


class base_RNN(nn.Module):
    def __init__(self, config, item_dim, sess_dim):
        super(base_RNN, self).__init__()

        self.config = config
        self.item_dim = item_dim
        self.sess_dim = sess_dim
        self.impressions_len = 25

        self.rnn_dim = config['rnn_dim']
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']

        self.lstm = nn.LSTM(input_size=self.sess_dim, hidden_size=self.rnn_dim, num_layers=self.num_layers, batch_first=True)

        self.item_embedding = nn.Linear(self.item_dim, self.embed_dim)
        self.linear = nn.Linear(self.embed_dim + self.rnn_dim, self.hidden_dim)
        self.scores = nn.Linear(self.hidden_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def label_idx(self, labels, item_idx):
        label_idx = []
        for i in range(len(labels)):
            label = str(labels[i])
            try:
                label_idx.append(item_idx[i].index(label))
            except:
                ## todo : ValueError confirm
                label_idx.append(0)
        return label_idx

    def BPR_loss(self, scores, label_idx):
        positive_scores = torch.zeros(scores.size(0), 1).to(torch.device("cuda" if use_cuda else "cpu"))
        for i in range(len(label_idx)):
            positive_scores[i] = scores[i, label_idx[i]]
        subtracts = positive_scores.expand(scores.size(0), self.impressions_len) - scores
        loss = - torch.mean(torch.log(self.sigmoid(subtracts) + 1e-6), 1)
        return loss

    def TOP1_loss(self, scores, label_idx):
        positive_scores = torch.zeros(scores.size(0), 1).to(torch.device("cuda" if use_cuda else "cpu"))
        for i in range(len(label_idx)):
            positive_scores[i] = scores[i, label_idx[i]]
        subtracts = self.sigmoid(scores - positive_scores.expand(scores.size(0), self.impressions_len))
        squares = self.sigmoid(scores**2)
        loss = torch.mean(subtracts + squares, 1)
        return loss

    def forward(self, sess_vectors, item_vectors, labels, item_idx, loss_type):
        """
        :param sess_vectors: PackedSequnce, [batch_size, sum of sequence lengths of batch]
        :param item_vectors: float tensor, [batch_size, impressions length(25), item_dim]
        :param labels: list, [batch_size]
        :param item_idx: lists contain item indexes of each session
        :param loss_type: top1, bpr, test
        :return: item scores, loss
        """

        h0 = torch.zeros(self.num_layers, item_vectors.size(0), self.rnn_dim).to(torch.device("cuda" if use_cuda else "cpu"))
        c0 = torch.zeros(self.num_layers, item_vectors.size(0), self.rnn_dim).to(torch.device("cuda" if use_cuda else "cpu"))

        packed_out, (ht, ct) = self.lstm(sess_vectors, (h0, c0))

        out, lengths = pad_packed_sequence(packed_out, batch_first=True)
        last_hidden = ht[-1].unsqueeze(-1)
        last_hidden = last_hidden.expand(last_hidden.size(0), last_hidden.size(1), self.impressions_len).permute(0,2,1)

        item_embed = self.item_embedding(item_vectors)

        item_sess = torch.cat((item_embed, last_hidden), dim=2)
        item_sess = self.linear(item_sess)
        scores = self.scores(item_sess).squeeze()

        if loss_type == 'bpr':
            label_idx = self.label_idx(labels, item_idx)
            loss = torch.mean(self.BPR_loss(scores, label_idx))
        elif loss_type == 'top1':
            label_idx = self.label_idx(labels, item_idx)
            loss = torch.mean(self.TOP1_loss(scores, label_idx))
        elif loss_type == 'test':
            loss = torch.zeros(1).to(torch.device("cuda" if use_cuda else "cpu"))
        else: raise NotImplementedError
        return scores, loss


if __name__ == "__main__":
    config = HParams.load("utils/hparams.yaml")
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 4
    max_len = 25
    sess_dim = 10
    item_dim = 157

    # loss_type = 'top1'
    # loss_type = 'bpr'
    loss_type = 'test'

    labels = [1, 2, 3, 4]
    item_idx = [['1','2','3','4'],['1','2','3','4'],['1','2','3','4'],['1','2','3','4']]
    session_vectors = torch.randn(batch_size, sess_dim, requires_grad=True).to(device)
    item_vectors = torch.randn(batch_size, max_len, item_dim, requires_grad=True).to(device)

    model = base_RNN(config.model, item_dim, sess_dim).to(device)

    scores, loss = model(session_vectors, item_vectors, labels, item_idx, loss_type)
    print(scores)
    print(loss)
