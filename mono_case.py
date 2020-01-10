from data.utils import load_data_torch
import pickle
from torch.nn import Module
from src.utils import *
from src.layers import *
from matplotlib import pyplot as plt
import sys
import time
import os

with open('./TIP/data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)


EPOCH_NUM = 10
feed_dict = load_data_torch("./TIP/data/", et_list, mono=True)
data = Data.from_dict(feed_dict)

n_drug, n_drug_feat = data.d_feat.shape
n_prot, n_prot_feat = data.p_feat.shape
data.train_idx, data.train_et, data.train_range, data.test_idx, data.test_et, data.test_range = process_edges(data.dd_edge_index)
n_mono = n_drug_feat - n_drug

threshold = 50
tmp = data.d_feat.to_dense()[:, n_drug:]
mask = tmp.sum(dim=0) < threshold
tmp[:, mask] = False
tmp = torch.nonzero(tmp[:, ])
node_index, node_type = tmp[:, 0], tmp[:, 1]


class mono_wraper(object):
    '''
        The number of drugs that have connection with proteins is only 200+. For each mono-side effect,
        the number of training instance must be smaller than this number
    '''

    def __init__(self, d_feat, d_mask, threshold):
        super(mono_wraper, self).__init__()

        self.n_drug = d_feat.shape[0]
        self.n_mono = d_feat.shape[1] - self.n_drug
        self.d_mask = d_mask

        self.monos = d_feat.to_dense()[:, self.n_drug:]
        self.monos[self.d_mask, :] = False

        self.mono_mask = self.monos.sum(dim=0) < threshold
        self.monos[:, self.mono_mask] = False

        tmp = torch.nonzero(self.monos)
        self.pos_node_index, self.pos_node_type = tmp[:, 0], tmp[:, 1]

    def pos_index_and_type(self):
        return self.pos_node_index, self.pos_node_type

    def neg_sampling(self):
        pass




class PP(torch.nn.Module):

    def __init__(self, in_dim, nhid_list):
        super(PP, self).__init__()
        self.out_dim = nhid_list[-1]
        self.embed = torch.nn.Linear(in_dim, nhid_list[0], bias=False)
        self.conv_list = [GCNConv(nhid_list[i], nhid_list[i+1], cached=True) for i in range(len(nhid_list)-1)]

    def forward(self, x, pp_edge_index, edge_weight):
        x = self.embed(x)
        for net in self.conv_list[:-1]:
            x = net(x, pp_edge_index, edge_weight)
            x = F.relu(x, inplace=True)
        x = net(x, pp_edge_index, edge_weight)
        return x


class PD(torch.nn.Module):

    def __init__(self, protein_dim, drug_dim, n_drug):
        super(PD, self).__init__()
        self.p_dim = protein_dim
        self.d_dim = drug_dim
        self.n_drug = n_drug
        self.conv = GCNConv(protein_dim, drug_dim, cached=True)

    def forward(self, x, pd_edge_index):
        return self.conv(x, pd_edge_index)[:self.n_drug, :]


class MultiClassScore(torch.nn.Module):
    def __init__(self, indim, nhid1, nhid2, num_nt):
        super(MultiClassScore, self).__init__()

        self.indim = indim
        self.nhid1 = nhid1
        self.nhid2 = nhid2

        self.linear1 = torch.nn.Linear(indim, nhid1)
        self.linear2 = torch.nn.Linear(nhid1, nhid2)
        self.weight = Param(torch.Tensor(num_nt, nhid2))

        self.reset_parameters()

    def forward(self, z, node_index, node_type):
        z = self.linear1(z)
        z = torch.nn.ReLU(z, inplace=True)
        z = self.linear2(z)
        z = (z[node_index, :] * self.weight[node_type, :]).sum(axis=1)
        return z

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.nhid2))


class Model(torch.nn.Module):

    def __init__(self, pp, pd, mc):
        super(Model, self).__init__()
        self.pp = pp
        self.pd = pd
        self.mc = mc


device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_name)
device = torch.device(device_name)


nhids_gcn = [32, 16]
drug_dim = [16]
nhids_nn = [16, 16]


pp = PP(n_prot, nhids_gcn).to(device)
pd = PD(nhids_gcn[-1], drug_dim, n_drug).to(device)
mc = MultiClassScore(drug_dim, nhids_nn[0], nhids_nn[2], n_mono).to(device)
model = Model(pp, pd, mc)




