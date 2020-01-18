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

print(os.path.abspath(os.getcwd()))

root = '.'

with open(root + '/data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)


out_dir = root + '/evaluation/exp_out/'

et_list = et_list
EPOCH_NUM = 50
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


nhids_gcn = [64, 64, 64]
prot_out_dim = sum(nhids_gcn)
drug_dim = 128


pp = PP(n_prot, nhids_gcn)
pd = PD(prot_out_dim, drug_dim, n_drug)
mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, n_et)
model = Model(pp, pd, mip).to('cpu')

name = 'poly-' + str(nhids_gcn) + '-' + str(drug_dim)
model.load_state_dict(torch.load(out_dir + name + '-model.pt'))

pp_static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)
pd_static_edge_weights = torch.ones((data.pd_index.shape[1])).to(device)

pre_mask = Pre_mask(data.pp_index.shape[1] // 2, data.pd_index.shape[1]).to(device)
data = data.to(device)
model = model.to(device)

for gcn in model.pp.conv_list:
    gcn.cached = False
model.pd.conv.cached = False


test_neg_index = typed_negative_sampling(data.test_idx, n_drug, data.test_range).to(device)
model.eval()























record = np.zeros((4, n_et))
for j in range(4):
    x = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)

    x.data[:, sum(nhids_gcn[:j]):] *= 0

    z = model.pd(x, data.pd_index)
    pos_score = model.mip(z, data.test_idx, data.test_et)
    neg_score = model.mip(z, test_neg_index, data.test_et)
    for i in range(data.test_range.shape[0]):
        [start, end] = data.test_range[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[j, i] = auprc_auroc_ap(target, score)[0]
print(record.mean(axis=1))



















record = np.zeros((3, n_et))
for j in range(1, 4):
    x = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)

    x.data[:, sum(nhids_gcn[:j]):] *= 0

    z = model.pd(x, data.pd_index)
    pos_score = model.mip(z, data.test_idx, data.test_et)
    neg_score = model.mip(z, test_neg_index, data.test_et)
    for i in range(data.test_range.shape[0]):
        [start, end] = data.test_range[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[j-1, i] = auprc_auroc_ap(target, score)[0]


et_index = np.array(final_et_list).reshape(-1, 1)
combine = np.concatenate([et_index, np.array(n_edges_per_type).reshape(-1, 1), record.T], axis=1)
df = pds.DataFrame(combine, columns=['side_effect', 'n_instance', 'auprc-1', 'auprc-2', 'auprc-3'])
df.astype({'side_effect': 'int32'})
df.to_csv('./' + name + '-layerwise.csv')




optimizer = torch.optim.Adam(pre_mask.parameters(), lr=0.01)
fake_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

'''
side effect id: 757
new index: 695
n_instance: 273
auprc: 0th: 92; 1st: 95; 2nd: 97;
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
