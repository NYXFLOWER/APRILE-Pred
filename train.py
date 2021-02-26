# %%
# from PoSePath.src.layers import *
from operator import mod
from src.layers import *
from src.utils import process_edges, auprc_auroc_ap
import pandas as pd
import pickle
import os

#******************** Hyperparameter Setting *************************
sp_rate = 0.8           # dataset split rate for traing and testing
nhids_gcn = [64, 32, 32]
drug_dim = 128
#*********************************************************************

# set working directory 
root = os.path.abspath(os.getcwd())
data_dir = root + '/data/'

# load data
with open(data_dir + 'tipexp_data.pkl', 'rb') as f:
    data = pickle.load(f)

# identify device
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"==== DEVICE: {device_name} ====")
device = torch.device(device_name)

# split train and test dataset
data.train_idx, data.train_et, data.train_range, \
data.test_idx, data.test_et, data.test_range = process_edges(data.final_dd_edge_index, 0.9)

# generate negative samples for testing
test_neg_index = typed_negative_sampling(data.test_idx, data.n_drug, data.test_range).to(device)

#******************** Build Model ********************
prot_out_dim = sum(nhids_gcn)

pp = PP(data.n_prot, nhids_gcn)
pd = PD(prot_out_dim, drug_dim, data.n_drug)
mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, data.n_et)
model = Model(pp, pd, mip)

model = model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)

train_record = {}
test_record = {}
train_out = {}
test_out = {}


def record_eva(mod: str, pos_score: torch.Tensor, neg_score: torch.Tensor) -> np.array:
    assert mod in {'train', 'test'}, "'idx' should in {'train', 'test'}"
    
    record = np.zeros((3, data.n_et))
    
    for i in range(getattr(data, f'{mod}_range').shape[0]):
        [start, end] = getattr(data, f'{mod}_range')[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)
    
    return record


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

    record = record_eva('train', pos_score, neg_score)

    train_record[epoch] = record
    [auprc, auroc, ap] = record.mean(axis=1)
    train_out[epoch] = [auprc, auroc, ap]

    print('{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}'
          .format(epoch, loss.tolist(), auprc, auroc, ap))

    return z, loss


def test(z):
    model.eval()

    pos_score = model.mip(z, data.test_idx, data.test_et)
    neg_score = model.mip(z, test_neg_index, data.test_et)

    return record_eva('test', pos_score, neg_score)


def layer_wise_eva():
    model.eval()
    n_average = 200
    n_layer = 3

    record = np.zeros((3, data.n_et))
    for ave in range(n_average):
        train_neg_index = typed_negative_sampling(data.train_idx, data.n_drug, data.train_range)

        for j in range(1, n_layer+1):
            x = model.pp(data.p_feat, data.pp_index, data.pp_static_edge_weights)

            # remain the (previous and) current layer(s)
            x.data[:, sum(nhids_gcn[:j]):] *= 0     

            z = model.pd(x, data.pd_index, data.pd_static_edge_weights)
            pos_score = model.mip(z, data.train_idx, data.train_et)
            neg_score = model.mip(z, train_neg_index, data.train_et)

            for i in range(data.train_range.shape[0]):
                [start, end] = data.train_range[i]

                p_s = pos_score[start: end]
                n_s = neg_score[start: end]

                pos_target = torch.ones(p_s.shape[0])
                neg_target = torch.zeros(n_s.shape[0])

                score = torch.cat([p_s, n_s])
                target = torch.cat([pos_target, neg_target])

                record[j - 1, i] += auprc_auroc_ap(target, score)[0]

    record = record / n_average

    # construct table and write to file
    et_index = np.array(range(data.n_et))
    df_data = np.concatenate([et_index, np.array(data.n_edge_per_type).reshape(-1, 1), record.T], axis=1)
    df = pd.DataFrame(df_data, columns=['side_effect', 'n_instance', 'auprc-0', 'auprc-1', 'auprc-2'])
    
    # //TODO
    

            



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


















