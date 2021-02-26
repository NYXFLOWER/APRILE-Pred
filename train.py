# %%
# from PoSePath.src.layers import *
from .src.layers import *
from .src.utils import process_edges

import pickle
import os

#******************** Hyperparameter Setting *************************
sp_rate = 0.8           # dataset split rate for traing and testing
nhids_gcn = [64, 32, 32]
drug_dim = 128


# -------------- set working directory --------------
root = os.path.abspath(os.getcwd())
data_dir = root + '/data/'

# -------------- load data --------------
with open(data_dir + 'tipexp_data.pkl', 'rb') as f:
    data = pickle.load(f)

# -------------- identify device --------------
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print("==== DEVICE: {} ====".format(device_name))
device = torch.device(device_name)

# split train and test dataset
data.train_idx, data.train_et, data.train_range, \
data.test_idx, data.test_et, data.test_range = process_edges(data.final_dd_edge_index, 0.9)

#******************** Build Model ********************
prot_out_dim = sum(nhids_gcn)

pp = PP(data.n_prot, nhids_gcn)
pd = PD(prot_out_dim, drug_dim, data.n_drug)
mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, n_et)
model = Model(pp, pd, mip)

model = model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)





# %%
