from data.utils import load_data_torch
import pickle
from torch.nn import Module
from src.utils import *
from src.layers import *
from matplotlib import pyplot as plt
import sys
import time
import pandas as pds
import os

root = '..'

with open(root + '/data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)


out_dir = root + '/exp_out/'

et_list = et_list
EPOCH_NUM = 100
feed_dict = load_data_torch(root + "/data/", et_list, mono=True)
data = Data.from_dict(feed_dict)

n_drug, n_drug_feat = data.d_feat.shape
n_prot, n_prot_feat = data.p_feat.shape

data.pp_index = get_edge_index_from_coo(data.pp_adj, True)
data.pd_index = get_edge_index_from_coo(data.dp_adj, False)[[1, 0], :]



# Reindexing drugs and edge types (side effects)
drug_dict = {}
for new_id, old_id in enumerate(drug_list_with_protein_links):
    drug_dict[old_id] = new_id
ufunc = np.frompyfunc(lambda x: drug_dict[x], 1, 1)
data.pd_index[1, :] = torch.from_numpy(ufunc(data.pd_index[1, :].numpy()).astype(np.int64))
final_dd_edge_index = []
final_et_list = []
for et, idx in zip(et_list, data.dd_edge_index):
    mask = get_indices_mask(idx, drug_list_with_protein_links)
    if mask.numpy().sum() > 200:
        final_et_list.append(et)
        tmp = idx[:, mask]
        final_dd_edge_index.append(torch.from_numpy(ufunc(tmp.numpy()).astype(np.int64)))
data.final_dd_edge_index = final_dd_edge_index
n_edges_per_type = [l.shape[1] for l in data.final_dd_edge_index]
n_drug = drug_list_with_protein_links.shape[0]
# Reindexing finished


data.train_idx, data.train_et, data.train_range, \
data.test_idx, data.test_et, data.test_range = process_edges(data.final_dd_edge_index)

n_et = len(data.test_range)
print('{}'.format(n_et) + ' final side effects.')


device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_name)
device = torch.device(device_name)


nhids_gcn = [32, 32, 32, 32, 32, 32]
prot_out_dim = sum(nhids_gcn)
drug_dim = 128


pp = PP(n_prot, nhids_gcn)
pd = PD(prot_out_dim, drug_dim, n_drug)
mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, n_et)
model = Model(pp, pd, mip).to('cpu')

name = 'poly-' + str(nhids_gcn) + '-' + str(drug_dim)
model.load_state_dict(torch.load(out_dir + name + '-model.pt'))



class Pre_mask(torch.nn.Module):
    def __init__(self, pp_n_link, pd_n_link):
        super(Pre_mask, self).__init__()
        self.pp_weight = Parameter(torch.Tensor(pp_n_link))
        self.pd_weight = Parameter(torch.Tensor(pd_n_link))
        self.reset_parameters()

    def reset_parameters(self):
        self.pp_weight.data.normal_(mean=0, std=1)
        self.pd_weight.data.normal_(mean=0, std=1)

pp_static_edge_weights = torch.zeros((data.pp_index.shape[1])).to(device)
pd_static_edge_weights = torch.zeros((data.pd_index.shape[1])).to(device)

pre_mask = Pre_mask(data.pp_index.shape[1] // 2, data.pd_index.shape[1]).to(device)
data = data.to(device)
model = model.to(device)

for gcn in model.pp.conv_list:
    gcn.cached = False
model.pd.conv.cached = False

optimizer = torch.optim.Adam(pre_mask.parameters(), lr=0.01)
fake_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

'''
side effect id: 458
index: 434
n_instance: 748
auprc: 97
'''

pre_mask.reset_parameters()
for i in range(100):

    model.train()
    optimizer.zero_grad()
    fake_optimizer.zero_grad()

    half_mask = torch.sigmoid(pre_mask.pp_weight)
    pp_mask = torch.cat([half_mask, half_mask])
    pd_mask = torch.sigmoid(pre_mask.pd_weight)

    z = model.pp(data.p_feat, data.pp_index, pp_mask)
    z = model.pd(z, data.pd_index, pd_mask)

    # z = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)
    # z = model.pd(z, data.pd_index, pd_static_edge_weights)

    e1, e2, et = 38, 179, 434
    P = torch.sigmoid((z[e1, :] * z[e2, :] * model.mip.weight[et, :]).sum())
    loss = -torch.log(P) + (pp_mask + pp_mask * (1 - pp_mask)).sum() + (pd_mask + pd_mask * (1 - pd_mask)).sum()

    loss.backward()
    optimizer.step()

    print(i, ' ', loss.tolist(), ' ', P.tolist(), ' ', pp_mask.sum().tolist())
