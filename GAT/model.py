import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import torch.nn as nn
import numpy as np
from dgl.nn import GATConv


class MyLoss_Pretrain(torch.nn.Module):
    def __init__(self):
        super(MyLoss_Pretrain, self).__init__()
        return

    def forward(self, pred, tar):
        kg_pred_max = torch.max(pred, dim=1)[0].view(-1, 1)
        kg_pred_log_max_sum = torch.log(torch.sum(torch.exp(pred-kg_pred_max), dim=1)).view(-1, 1)
        kg_pred_log_softmax = pred - kg_pred_max - kg_pred_log_max_sum
        loss_kge = - kg_pred_log_softmax[tar==True].sum()
        return loss_kge




"""
code for load pretrained embeddings, you can choose whether to update embedding params in training with freeze param

        if params['pretrain'] == 'true':
            print('Loading pretrained weights....')
            pretrain_emb = np.load('./pretrain_emb/ER_' + args.dataset + '_TuckER' + '_' + str(edim) + '.npz')
            params['E_pretrain'] = torch.from_numpy(pretrain_emb['E_pretrain']).to(device)
            params['R_pretrain'] = torch.from_numpy(pretrain_emb['R_pretrain']).to(device)
            
            
            
        if kwargs['pretrain'] == 'true':
            freeze_bool = True if kwargs['freeze'] == 'true' else False
            self.E = nn.Embedding.from_pretrained(kwargs['E_pretrain'], freeze=freeze_bool)
            self.R = nn.Embedding.from_pretrained(kwargs['R_pretrain'], freeze=freeze_bool)
        else:
            self.E = torch.nn.Embedding(len(d.ent2id), edim)
            self.R = torch.nn.Embedding(len(d.rel2id), edim)
            self.init()
"""

class TuckER(torch.nn.Module):
    def __init__(self, d, d1, **kwargs):
        super(TuckER, self).__init__()
        ne, nr = len(d.ent2id), len(d.rel2id)
        edim, d2 = d1, d1
        device = kwargs['device']
        self.E = torch.nn.Embedding(ne, d1)
        self.R = torch.nn.Embedding(nr, d2)
        self.init()
        self.gcn_layers = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.n_layer = kwargs['n_layer']
        for i in range(0, kwargs['n_layer']-1):
            self.gcn_layers.append(GATConv(edim,edim,kwargs['num_heads'],allow_zero_in_degree=True))
            self.dropout.append(nn.Dropout(kwargs['dropout']))
        self.gcn_layers.append(GATConv(edim,edim,kwargs['num_heads'],allow_zero_in_degree=True))
        self.dropout.append(nn.Dropout(kwargs['dropout']))

        self.loss = MyLoss_Pretrain()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, g, h_idx):
        E_feat = self.E.weight
        for i in range(self.n_layer-1):
            E_feat = self.gcn_layers[i](g, E_feat)
            E_feat = self.dropout[i](E_feat)
            E_feat = torch.mean(E_feat,dim=1)
            E_feat = F.tanh(E_feat)
        E_feat = self.gcn_layers[-1](g, E_feat)
        E_feat = self.dropout[-1](E_feat)
        E_feat = torch.mean(E_feat,dim=1)
        E_feat = torch.tanh(E_feat)

        h=E_feat[h_idx] # bs*edim

        pred = torch.mm(h, E_feat.transpose(1, 0)) # bs*ne
        return pred
    
    def encode(self,g):
        E_feat = self.E.weight
        for i in range(self.n_layer-1):
            E_feat = self.gcn_layers[i](g, E_feat)
            # E_feat = self.dropout[i](E_feat)
            E_feat = torch.mean(E_feat,dim=1)
            E_feat = F.tanh(E_feat)
        E_feat = self.gcn_layers[-1](g, E_feat)
        # E_feat = self.dropout[-1](E_feat)
        E_feat = torch.mean(E_feat,dim=1)
        E_feat = torch.tanh(E_feat)
        return E_feat