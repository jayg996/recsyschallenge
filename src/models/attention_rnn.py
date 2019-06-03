from utils.hparams import HParams
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from utils.pytorch_utils import LayerNorm
import torch.nn.functional as F
import time

use_cuda = torch.cuda.is_available()


class attn_RNN(nn.Module):
    def __init__(self, config, item_dim, sess_dim):
        super(attn_RNN, self).__init__()

        self.config = config
        self.item_dim = item_dim
        self.sess_dim = sess_dim
        self.impressions_len = 25

        self.rnn_dim = config['rnn_dim']
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.price_dim = config['price_dim']
        self.num_layers = config['num_layers']
        self.drop_out = config['drop_out']

        self.lstm = nn.LSTM(input_size=self.sess_dim, hidden_size=self.rnn_dim, num_layers=self.num_layers, batch_first=True)

        self.item_embedding = nn.Linear(self.item_dim, self.embed_dim)
        self.price_embedding = nn.Linear(1, self.price_dim)
        self.linear = nn.Linear(self.embed_dim + self.price_dim + self.rnn_dim, self.hidden_dim)
        self.scores = nn.Linear(self.hidden_dim, 1)

        self.query_linear = nn.Linear(self.embed_dim + self.price_dim, self.embed_dim)
        self.key_linear = nn.Linear(self.rnn_dim, self.embed_dim)
        self.value_linear = nn.Linear(self.rnn_dim, self.embed_dim)
        self.context_linear = nn.Linear(self.embed_dim, self.embed_dim)

        self.query_scale = self.embed_dim ** -0.5

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.drop_out)

        self.layer_norm_price = LayerNorm(self.price_dim)
        self.layer_norm_item = LayerNorm(self.embed_dim)
        self.layer_norm_sess = LayerNorm(self.rnn_dim)
        self.layer_norm_out = LayerNorm(self.embed_dim)

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

    def forward(self, sess_context_vectors, item_vectors, labels, item_idx, prices, loss_type):
        """
        :param sess_context_vectors: PackedSequnce, [sum of sequence lengths of batch, session and context dimension(170)]
        :param item_vectors: float tensor, [batch_size, impressions length(25), item_dim]
        :param labels: list, [batch_size]
        :param item_idx: lists contain item indexes of each session
        :param prices: float tensor, [batch_size, impressions length]
        :param loss_type: top1, bpr, test
        :return: item scores, loss
        """

        h0 = torch.zeros(self.num_layers, item_vectors.size(0), self.rnn_dim).to(torch.device("cuda" if use_cuda else "cpu"))
        c0 = torch.zeros(self.num_layers, item_vectors.size(0), self.rnn_dim).to(torch.device("cuda" if use_cuda else "cpu"))

        packed_out, (ht, ct) = self.lstm(sess_context_vectors, (h0, c0))

        out, lengths = pad_packed_sequence(packed_out, batch_first=True)
        out = self.layer_norm_sess(out)
        # out = out.exapand()

        item_embed = self.item_embedding(item_vectors)
        item_embed = self.layer_norm_item(item_embed)
        price_embed = self.price_embedding(prices.unsqueeze(-1))
        price_embed = self.layer_norm_price(price_embed)

        item_price = torch.cat((item_embed, price_embed), dim=2)

        queries = self.query_linear(item_price)
        queries *= self.query_scale

        keys = self.key_linear(out)
        values = self.value_linear(out)

        logits = torch.matmul(queries, keys.permute(0,2,1))
        weights = F.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        contexts = torch.matmul(weights, values)
        outputs = self.context_linear(contexts)
        outputs = self.layer_norm_out(outputs)

        item_sess = torch.cat((item_price, outputs), dim=2)
        item_sess = self.linear(item_sess)
        item_sess = F.leaky_relu(item_sess)
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
