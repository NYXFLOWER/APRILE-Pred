# %%
# from PoSePath.src.layers import *
from operator import mod
from src.layers import *
from src.utils import process_edges, auprc_auroc_ap
import pandas as pds
import pickle
import time
import os
import sys


#******************** Hyperparameter Setting *************************
# sp_rate = 0.8           # dataset split rate for traing and testing
# nhids_gcn = [64, 32, 32]
# drug_dim = 128
# EPOCH_NUM = 2

sp_rate = float(sys.argv[1])
nhids_gcn = [int(sys.argv[i]) for i in [2, 3, 4]]
drug_dim = int(sys.argv[5])
EPOCH_NUM = int(sys.argv[6])
torch.manual_seed(sys.argv[7])
#*********************************************************************

# set working directory 
root = os.path.abspath(os.getcwd())
data_dir = os.path.join(root, 'data')
out_dir = os.path.join(root, 'evaluation/new_out')
name = f'{nhids_gcn}-{drug_dim}-{sp_rate}-{EPOCH_NUM}-{sys.argv[7]}'

# load data
with open(os.path.join(data_dir, 'tipexp_data.pkl'), 'rb') as f:
    data = pickle.load(f)

# init output data
train_out = np.zeros((EPOCH_NUM, 3))
test_out = np.zeros((EPOCH_NUM, 3))

# identify device
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

print("====================== Data Preparing ======================\n")
print(f"Data split: {int(sp_rate*100)}\% data for trining")
# split train and test dataset
data.train_idx, data.train_et, data.train_range, \
data.test_idx, data.test_et, data.test_range = process_edges(data.final_dd_edge_index, sp_rate)
print(f"--> positive training samples: {data.train_et.shape[0]}")
print(f"--> positive testing  samples: {data.test_et.shape[0]}\n")

# generate negative samples for testing
test_neg_index = typed_negative_sampling(data.test_idx, data.n_drug, data.test_range).to(device)
print('Negative testing samples are generated')

print("====================== Model Setting ======================\n")
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
print(model)


def evaluate(mod: str, pos_score: torch.Tensor, neg_score: torch.Tensor) -> np.array:
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


def train() -> tuple:

    model.train()
    optimizer.zero_grad()

    z = model.pp(data.p_feat, data.pp_index, static_edge_weights)
    z = model.pd(z, data.pd_index)

    pos_index = data.train_idx
    neg_index = typed_negative_sampling(data.train_idx, data.n_drug, data.train_range).to(device)

    pos_score = model.mip(z, pos_index, data.train_et)
    neg_score = model.mip(z, neg_index, data.train_et)

    pos_loss = -torch.log(pos_score + EPS).mean()
    neg_loss = -torch.log(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss

    loss.backward()

    optimizer.step()

    record = evaluate('train', pos_score, neg_score)

    return z, loss, record


def test(z) -> np.array:
    model.eval()

    pos_score = model.mip(z, data.test_idx, data.test_et)
    neg_score = model.mip(z, test_neg_index, data.test_et)

    return evaluate('test', pos_score, neg_score)


def layer_wise_eva() -> np.array:
    model.eval()
    
    n_average = 200
    n_layer = 3

    record = np.zeros((3, data.n_et))
    for ave in range(n_average):
        train_neg_index = typed_negative_sampling(data.train_idx, data.n_drug, data.train_range)

        for j in range(1, n_layer+1):
            x = model.pp(data.p_feat, data.pp_index, static_edge_weights)

            # remain the (previous and) current layer(s)
            x.data[:, sum(nhids_gcn[:j]):] *= 0     

            z = model.pd(x, data.pd_index)
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

    return record

            
if __name__ == '__main__':
    print('====================== Model training ====================== \n')
    for epoch in range(EPOCH_NUM):
        time_begin = time.time()

        #********************** Train **********************
        z, loss, record_train = train()
        [auprc, auroc, ap] = record_train.mean(axis=1)
        train_out[epoch] = [auprc, auroc, ap]

        print(f'{epoch:3d}   loss:{loss.tolist():0.4f}   auprc:{auprc:0.4f}   auroc:{auroc:0.4f}   ap@50:{ap:0.4f}')

        #********************** Test **********************
        record_test = test(z)
        [auprc, auroc, ap] = record_test.mean(axis=1)
        test_out[epoch] = [auprc, auroc, ap]

        print(f'{epoch:3d}   loss:{loss.tolist():0.4f}   auprc:{auprc:0.4f}   auroc:{auroc:0.4f}   ap@50:{ap:0.4f}    time:{(time.time() - time_begin):0.1f}\n')

    # print('=================== Layer-wised evaluation =================== \n')
    # last_record = record_test.T
    # nlayer_record = layer_wise_eva().T
    
    # et_idx = np.array(range(data.n_et)).astype(np.int).reshape((-1, 1))
    # et_name = np.array([data.side_effect_idx_to_name[i] for i in range(data.n_et)]).astype(np.str).reshape((-1, 1))
    # n_instance = np.array(data.n_edges_per_type).astype(np.int).reshape((-1, 1))

    # df_data = np.concatenate([et_idx, et_name, n_instance, last_record, nlayer_record], axis=1)
    # df = pds.DataFrame(df_data, columns=['side_effect_idx', 'side_effect', 'n_instance', 'auprc', 'auroc', 'ap@50', 'auprc_layer-0', 'auprc_layer-1', 'auprc_layer-2'])

    print('====================== Results Saving ====================== \n')
    # save model
    torch.save(model.to('cpu').state_dict(), os.path.join(out_dir, name+'-model.pt'))
    print(f"The trained model is saved at epoch {EPOCH_NUM}")

    # save record
    with open(os.path.join(out_dir, name+'-record.pickle'), 'wb') as f:
        pickle.dump({'train_out': train_out, 'test_out': test_out}, f)
    print('The training and testing records are saved')

    with open(os.path.join(out_dir, name+'.txt'), 'w') as f:
        f.write("train: " + str(train_out[EPOCH_NUM-1]) + "\n")
        f.write("test : " + str(test_out[EPOCH_NUM-1]))

    # # save evaluation
    # df.to_csv(os.path.join(out_dir, name + '-record.csv'))
    # print('The evalution of the trained model is saved')

    print('======================== FINISHED ========================')



















# %%
