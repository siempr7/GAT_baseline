import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import torch.nn as nn
import numpy as np
from torch_geometric.nn import RGCNConv

class MyLoss_Pretrain(torch.nn.Module):
    def __init__(self):
        super(MyLoss_Pretrain, self).__init__()
        return

    def forward(self, pred, true):
        mask=(true != -1).clone().detach().to(torch.bool)
        squared_errors=(true[mask] - pred[mask]) ** 2
        loss= squared_errors.mean()
        return loss

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project=nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w=self.project(z).mean(0)
        beta=torch.softmax(w, dim=0)
        beta=beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)

class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, nrs, dropout, nreg, sub_layer, d):
        super(HANLayer, self).__init__()
        self.gnn_layers=nn.ModuleList()
        for i in range(num_meta_paths):
            for j in range(sub_layer-1):
                self.gnn_layers.append(RGCNConv(in_size, in_size, nrs[i]))
            self.gnn_layers.append(RGCNConv(in_size, out_size, nrs[i]))
        
        # self.semantic_attention=SemanticAttention(in_size=out_size)
        self.num_meta_paths=num_meta_paths
        self.dropout=dropout
        self.nreg=nreg
        self.sublayer=sub_layer

        self.query_projection = nn.Linear(d.metapath_emb.shape[1], out_size)
        self.mha = nn.MultiheadAttention(out_size, num_heads=1)

    def forward(self, gs, E, metapath_emb, ifdropout):
        semantic_embeddings = []
        for i,g in enumerate(gs):
            edge_index,eids = g[0],g[1]
            E_feat = E[eids]
            for j in range(self.sublayer):
                E_feat = self.gnn_layers[self.sublayer*i + j](E_feat, edge_index=edge_index)
                if ifdropout:
                    E_feat = F.dropout(E_feat, p=self.dropout)
                E_feat = F.relu(E_feat)
            semantic_embeddings.append(E_feat[:self.nreg])

        embeddings = torch.stack(semantic_embeddings, dim=1)  # nreg, n_meta_paths, out_size

        # query
        query = self.query_projection(metapath_emb) # n_meta_paths, out_size
        query = query[None, :, :].repeat(self.nreg, 1, 1) # nreg, n_meta_paths, out_size
        key = embeddings.permute(0, 1, 2) # nreg, n_meta_paths, out_size
        value = embeddings.permute(0, 1, 2) # nreg, n_meta_paths, out_size

        # attention
        scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(embeddings.shape[2]) # nreg, n_meta_paths, n_meta_paths
        # 取对角线元素q_i * k_i
        scores = scores[:, torch.arange(scores.shape[1]), torch.arange(scores.shape[2])] # nreg, n_meta_paths
        attention_weights = F.softmax(scores, dim=1) # nreg, n_meta_paths
        attention_weights = attention_weights[:, :, None].repeat(1, 1, embeddings.shape[2]) # nreg, n_meta_paths, out_size
        output = torch.sum(attention_weights * value, dim=1) # nreg, out_size

        # output = self.semantic_attention(embeddings) # nreg, out_size

        return output

class HAN(nn.Module):
    def __init__(self, d, **kwargs):
        super(HAN, self).__init__()
        num_meta_paths=len(d.relpaths)
        self.nmp=num_meta_paths
        self.nreg=d.nreg
        self.all_tasks = d.all_tasks
        ne=len(d.ent2id)
        nr=len(d.rel2id)
        nes=[len(v['ent2id']) for v in d.mp2data.values()]
        nrs=[len(v['rel2id']) for v in d.mp2data.values()]
        hidden_size=kwargs['hidden_size']

        self.E=torch.nn.Embedding(ne, kwargs['edim'])
        self.R=torch.nn.Embedding(nr, kwargs['edim'])
        self.init()

        self.rgcn=nn.ModuleList()
        for i in range(0, kwargs['sum_layer']):
            self.rgcn.append(RGCNConv(kwargs['edim'], kwargs['edim'], nr))
        self.dropout=kwargs['dropout']

        self.layers=HANLayer(num_meta_paths, kwargs['edim'], hidden_size, nrs, kwargs['dropout'], self.nreg, kwargs['sub_layer'], d)
        self.predict=nn.Linear(hidden_size, kwargs['edim'])

        self.cross_task_attention = SemanticAttention(in_size=kwargs['edim'])

        # downstream task
        self.edge_regressor=nn.Linear(2*kwargs['edim'], 1)

        self.loss=MyLoss_Pretrain()
    
    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, gs, edge_index, metapath_emb, all_tasks_emb):
        # RGCN
        E=self.E.weight
        for r in self.rgcn:
            E = r(E, edge_index=edge_index)
            E = F.dropout(E, p=self.dropout)
            E = torch.tanh(E)

        # metapaths
        h = self.layers.forward(gs, E, metapath_emb, ifdropout=True) # nreg, hidden_size
        E_reg = self.predict(h) # nreg, edim

        # combine with other task emb
        # input: nreg, n_tasks, edim
        # output: nreg, edim
        embeddings = [E_reg]
        for task in self.all_tasks:
            if all_tasks_emb[task+'_E_reg'] is not None:
                embeddings.append(all_tasks_emb[task+'_E_reg'])
        if len(embeddings) > 1:
            embeddings = torch.stack(embeddings, dim=1) # nreg, n_tasks, edim
            E_reg = self.cross_task_attention(embeddings) # nreg, edim

        E_reg = torch.cat((E_reg, E[:self.nreg]), dim=1)

        # 使用线性层
        pred = self.edge_regressor(E_reg)

        return pred
    

    def get_emb(self, gs, edge_index, metapath_emb):
        # RGCN
        E=self.E.weight
        for r in self.rgcn:
            E = r(E, edge_index=edge_index)
            E = torch.tanh(E)

        # metapaths
        h = self.layers.forward(gs, E, metapath_emb, ifdropout=False) # nreg, hidden_size
        E_reg = self.predict(h) # nreg, edim

        E_kg = E[:self.nreg]

        return E_reg, E_kg
