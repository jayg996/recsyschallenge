from hparams import HParams
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

use_cuda = torch.cuda.is_available()


class FM(nn.Module):
    def __init__(self, config, item_dim, sess_dim):
        super(FM, self).__init__()

        self.config = config
        self.item_dim = item_dim
        self.sess_dim = sess_dim
        self.max_len = 25
        self.Q = nn.Parameter(torch.Tensor(sess_dim, config['hidden_dim']))
        self.P = nn.Parameter(torch.Tensor(item_dim, config['hidden_dim']))
        self.b_p = nn.Parameter(torch.Tensor(sess_dim,1))
        self.b_q = nn.Parameter(torch.Tensor(item_dim,1))
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.Q)
        nn.init.xavier_normal_(self.P)
        nn.init.xavier_normal_(self.b_p)
        nn.init.xavier_normal_(self.b_q)

    def score(self, sess_vectors, item_vectors):
        SQ = torch.matmul(sess_vectors, self.Q).unsqueeze(1)
        IP = torch.transpose(torch.matmul(item_vectors, self.P), 1, 2)
        sum = torch.matmul(SQ, IP).squeeze(1)
        scores = sum + torch.matmul(sess_vectors, self.b_p).expand(sess_vectors.size(0), self.max_len) + torch.matmul(item_vectors,self.b_q).squeeze()
        return scores

    def label_idx(self, labels, item_idx):
        label_idx = []
        for i in range(len(labels)):
            label = str(labels[i])
            try:
                label_idx.append(item_idx[i].index(label))
            except:
                ## todo : ValueError confirm
                # print('clickout item is not in impressions : ' + label)
                label_idx.append(0)
        return label_idx

    def BPR_loss(self, scores, label_idx, item_idx):
        normal = True
        for i in item_idx:
            if len(i) != self.max_len:
                normal = False
        if normal:
            positive_scores = torch.zeros(scores.size(0), 1).to(torch.device("cuda" if use_cuda else "cpu"))
            for i in range(len(label_idx)):
                positive_scores[i] = scores[i, label_idx[i]]
            subtracts = positive_scores.expand(scores.size(0), self.max_len) - scores
            loss = - torch.mean(torch.log(self.sigmoid(subtracts) + 1e-6), 1)
        else:
            positive_scores = torch.zeros(scores.size(0), 1).to(torch.device("cuda" if use_cuda else "cpu"))
            for i in range(len(label_idx)):
                positive_scores[i] = scores[i, label_idx[i]]
            subtracts = positive_scores.expand(scores.size(0), self.max_len) - scores
            loss = torch.zeros(scores.size(0)).to(torch.device("cuda" if use_cuda else "cpu"))
            for i in range(len(item_idx)):
                for j in range(len(item_idx[i])):
                    loss[i] += - torch.log(self.sigmoid(subtracts[i,j]) + 1e-6)
                loss[i] /= len(item_idx[i])
        return loss

    def TOP1_loss(self, scores, label_idx, item_idx):
        normal = True
        for i in item_idx:
            if len(i) != self.max_len:
                normal = False
        positive_scores = torch.zeros(scores.size(0), 1).to(torch.device("cuda" if use_cuda else "cpu"))
        for i in range(len(label_idx)):
            positive_scores[i] = scores[i, label_idx[i]]
        subtracts = self.sigmoid(scores - positive_scores.expand(scores.size(0), self.max_len))
        squares = self.sigmoid(scores**2)
        if normal:
            loss = torch.mean(subtracts + squares, 1)
        else:
            loss = torch.zeros(scores.size(0)).to(torch.device("cuda" if use_cuda else "cpu"))
            for i in range(len(item_idx)):
                for j in range(len(item_idx[i])):
                    loss[i] += subtracts[i,j] + squares[i,j]
                loss[i] /= len(item_idx[i])
        return loss

    def forward(self, sess_vectors, item_vectors, labels, item_idx, loss_type='bpr'):
        scores = self.score(sess_vectors, item_vectors)
        if loss_type == 'bpr':
            label_idx = self.label_idx(labels, item_idx)
            loss = torch.mean(self.BPR_loss(scores, label_idx, item_idx))
        elif loss_type == 'top1':
            label_idx = self.label_idx(labels, item_idx)
            loss = torch.mean(self.TOP1_loss(scores, label_idx, item_idx))
        elif loss_type == 'test':
            loss = torch.zeros(1).to(torch.device("cuda" if use_cuda else "cpu"))
        else: raise NotImplementedError
        return scores, loss

if __name__ == "__main__":
    config = HParams.load("hparams.yaml")
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
    session_vectors = torch.randn(batch_size,sess_dim,requires_grad=True).to(device)
    item_vectors = torch.randn(batch_size, max_len, item_dim, requires_grad=True).to(device)

    model = FM(config.model, item_dim, sess_dim).to(device)

    scores, loss = model(session_vectors, item_vectors, labels, item_idx, loss_type)
    print(scores)
    print(loss)



