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


nhids_gcn = [128, 64, 64]
prot_out_dim = sum(nhids_gcn)
# prot_out_dim = 32
drug_dim = 128


pp = PP(n_prot, nhids_gcn)
pd = PD(prot_out_dim, drug_dim, n_drug)
mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, n_et)
model = Model(pp, pd, mip)

model = model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)

train_record = {}
test_record = {}
train_out = {}
test_out = {}


def train():

    model.train()
    optimizer.zero_grad()

    z = model.pp(data.p_feat, data.pp_index, static_edge_weights)
    z = model.pd(z, data.pd_index)


    pos_index = data.train_idx
    neg_index = typed_negative_sampling(data.train_idx, n_drug, data.train_range).to(device)

    pos_score = model.mip(z, pos_index, data.train_et)
    neg_score = model.mip(z, neg_index, data.train_et)

    pos_loss = -torch.log(pos_score + EPS).mean()
    neg_loss = -torch.log(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss

    loss.backward()


    optimizer.step()

    record = np.zeros((3, n_et))  # auprc, auroc, ap
    for i in range(data.train_range.shape[0]):
        [start, end] = data.train_range[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

    train_record[epoch] = record
    [auprc, auroc, ap] = record.mean(axis=1)
    train_out[epoch] = [auprc, auroc, ap]

    print('{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}'
          .format(epoch, loss.tolist(), auprc, auroc, ap))

    return z, loss

test_neg_index = typed_negative_sampling(data.test_idx, n_drug, data.test_range).to(device)


def test(z):
    model.eval()

    record = np.zeros((3, n_et))

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

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

    return record

if __name__ == '__main__':
    print('model training ...')
    for epoch in range(EPOCH_NUM):
        time_begin = time.time()

        z, loss = train()

        record_te = test(z)
        [auprc, auroc, ap] = record_te.mean(axis=1)

        print('{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}    time:{:0.1f}\n'
              .format(epoch, loss.tolist(), auprc, auroc, ap, (time.time() - time_begin)))

        test_record[epoch] = record_te
        test_out[epoch] = [auprc, auroc, ap]


    name = 'poly-' + str(nhids_gcn) + '-' + str(drug_dim)

    # save model
    torch.save(model.to('cpu').state_dict(), out_dir + name + '-model.pt')

    # save record
    last_record = test_record[EPOCH_NUM-1].T
    et_index = np.array(final_et_list).reshape(-1, 1)
    combine = np.concatenate([et_index, np.array(n_edges_per_type).reshape(-1, 1), last_record], axis=1)
    df = pds.DataFrame(combine, columns=['side_effect', 'n_instance', 'auprc', 'auroc', 'ap'])
    df.astype({'side_effect': 'int32'})
    df.to_csv(out_dir + name + '-record.csv')
