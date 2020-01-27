import sys
sys.path.extend(['..'])


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

root = '.'
out_dir = root + '/exp_out/'


class Pre_mask(torch.nn.Module):
    def __init__(self, pp_n_link, pd_n_link):
        super(Pre_mask, self).__init__()
        self.pp_weight = Parameter(torch.Tensor(pp_n_link))
        self.pd_weight = Parameter(torch.Tensor(pd_n_link))
        self.reset_parameters()

    def reset_parameters(self):
        # self.pp_weight.data.normal_(mean=0.5, std=0.01)
        # self.pd_weight.data.normal_(mean=0.5, std=0.01)
        self.pp_weight.data.fill_(0.99)
        self.pd_weight.data.fill_(0.99)



class Tip_explainer(object):
    def __init__(self, model, data, device, regulization=1):
        super(Tip_explainer, self).__init__()
        self.model = model
        self.data = data
        self.device = device
        self.regulization = regulization

    def explain(self, drug1, drug2, side_effect):

        data = self.data
        model = self.model
        device = self.device

        pre_mask = Pre_mask(data.pp_index.shape[1] // 2, data.pd_index.shape[1]).to(device)
        data = data.to(device)
        model = model.to(device)

        for gcn in self.model.pp.conv_list:
            gcn.cached = False
        self.model.pd.conv.cached = False
        self.model.eval()

        pp_static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)
        pd_static_edge_weights = torch.ones((data.pd_index.shape[1])).to(device)

        optimizer = torch.optim.Adam(pre_mask.parameters(), lr=0.01)
        fake_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        z = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)
        z = model.pd(z, data.pd_index, pd_static_edge_weights)

        P = torch.sigmoid((z[drug1, :] * z[drug2, :] * model.mip.weight[side_effect, :]).sum())
        print(P.tolist())


        tmp = 0.0
        pre_mask.reset_parameters()
        for i in range(300):
            model.train()
            optimizer.zero_grad()
            fake_optimizer.zero_grad()

            # half_mask = torch.sigmoid(pre_mask.pp_weight)
            half_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pp_weight)
            pp_mask = torch.cat([half_mask, half_mask])

            pd_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pd_weight)

            z = model.pp(data.p_feat, data.pp_index, pp_mask)

            # TODO:
            # z = model.pd(z, data.pd_index, pd_static_edge_weights)
            z = model.pd(z, data.pd_index, pd_mask)
            # TODO:

            P = torch.sigmoid((z[drug1, :] * z[drug2, :] * model.mip.weight[side_effect, :]).sum())

            # TODO:
            loss = self.regulization * torch.log(1 - P) + 0.5 * (pp_mask * (2 - pp_mask)).sum() + (pd_mask * (2 - pd_mask)).sum()
            # loss = - self.regulization * torch.log(P) + 0.5 * (pp_mask * (2 - pp_mask)).sum() + (pd_mask * (2 - pd_mask)).sum()
            # TODO:

            loss.backward()
            optimizer.step()

            print(i, ' ', loss.tolist(), ' ', P.tolist(), ' ', pp_mask.sum().tolist(), ' ', pd_mask.sum().tolist())

            if tmp == pp_mask.sum().tolist() + pd_mask.sum().tolist():
                break
            else:
                tmp = pp_mask.sum().tolist() + pd_mask.sum().tolist()

        pp_left_mask = (pp_mask > 0.2).detach().cpu().numpy()
        tmp = (data.pp_index[0, :] > data.pp_index[1, :]).detach().cpu().numpy()
        pp_left_mask = np.logical_and(pp_left_mask, tmp)

        pd_left_mask = (pd_mask > 0.2).detach().cpu().numpy()

        pp_left_index = data.pp_index[:, pp_left_mask].cpu().numpy()
        pd_left_index = data.pd_index[:, pd_left_mask].cpu().numpy()

        pp_left_weight = pp_mask[pp_left_mask].detach().cpu().numpy()
        pd_left_weight = pd_mask[pd_left_mask].detach().cpu().numpy()

        return pp_left_index, pp_left_weight, pd_left_index, pd_left_weight







with open(root + '/data/tipexp/tipexp_data.pkl', 'rb') as f:
    data = pickle.load(f)


device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_name)
device = torch.device(device_name)


nhids_gcn = [64, 32, 32]
prot_out_dim = sum(nhids_gcn)
drug_dim = 128
pp = PP(data.n_prot, nhids_gcn)
pd = PD(prot_out_dim, drug_dim, data.n_drug)
mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, data.n_et)
model = Model(pp, pd, mip).to('cpu')

name = 'poly-' + str(nhids_gcn) + '-' + str(drug_dim)
model.load_state_dict(torch.load(out_dir + name + '-model.pt'))

exp = Tip_explainer(model, data, device)

drug1, drug2, side_effect = 88, 95, 846
result = exp.explain(drug1, drug2, side_effect)

print(result)



# pre_mask = Pre_mask(data.pp_index.shape[1] // 2, data.pd_index.shape[1]).to(device)
# data = data.to(device)
# model = model.to(device)
#
# for gcn in model.pp.conv_list:
#     gcn.cached = False
# model.pd.conv.cached = False

# test_neg_index = typed_negative_sampling(data.test_idx, data.n_drug, data.test_range).to(device)
# model.eval()
#
#
# optimizer = torch.optim.Adam(pre_mask.parameters(), lr=0.01)
# fake_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# z = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)
# z = model.pd(z, data.pd_index, pd_static_edge_weights)
# predict = lambda e1, e2, et: float(torch.sigmoid((z[e1, :] * z[e2, :] * model.mip.weight[et, :]).sum()))
#
# e1, e2, et = 88, 95, 846
# P = torch.sigmoid((z[e1, :] * z[e2, :] * model.mip.weight[et, :]).sum())
# print(P.tolist())
#
# '''
# side effect index: 1049 (Breast inflammation)
# index in code: 846
# n_instance: 263
#
# drug1: 22
# drug2: 134
# index in code: 846
# prob: 87.1%  97.1%  97.4%
#
# '''
#
#
# pre_mask.reset_parameters()
# for i in range(300):
#
#     model.train()
#     optimizer.zero_grad()
#     fake_optimizer.zero_grad()
#
#     # half_mask = torch.sigmoid(pre_mask.pp_weight)
#     half_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pp_weight)
#     pp_mask = torch.cat([half_mask, half_mask])
#
#     pd_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pd_weight)
#
#
#     z = model.pp(data.p_feat, data.pp_index, pp_mask)
#     z = model.pd(z, data.pd_index, pd_static_edge_weights)
#     # z = model.pd(z, data.pd_index, pd_mask)
#
#     P = torch.sigmoid((z[e1, :] * z[e2, :] * model.mip.weight[et, :]).sum())
#
#     # loss = torch.log(1 - P) + 0.5 * (pp_mask * (2 - pp_mask)).sum() + (pd_mask * (2 - pd_mask)).sum()
#     loss = - 3 * torch.log(P) + 0.5 * (pp_mask * (2 - pp_mask)).sum() + (pd_mask * (2 - pd_mask)).sum()
#
#
#
#     loss.backward()
#     optimizer.step()
#
#     print(i, ' ', loss.tolist(), ' ', P.tolist(), ' ', pp_mask.sum().tolist(), ' ', pd_mask.sum().tolist())


# tmp = data.pp_index[:, data.pp_index[0, :] > data.pp_index[1, :]]
# print(tmp[:, pp_mask[data.pp_index[0, :] > data.pp_index[1, :]].detach().cpu().numpy() > 0.1].cpu().numpy())
# print(data.pd_index[:, pd_mask.detach().cpu().numpy() > 0.1].cpu().numpy())

