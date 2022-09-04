import torch
from torch import nn
import torch.nn.functional as F
# from nnet.spectral import SNLinear


class RGCN_Layer(nn.Module):
    """ A Relation GCN module operated on documents graphs. """

    def __init__(self, in_dim, mem_dim, num_layers, relation_cnt=5):
        super().__init__()
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.relation_cnt = relation_cnt
        self.in_dim = in_dim
        self.in_drop = nn.Dropout(0.2)
        self.gcn_drop = nn.Dropout(0.2)
        self.W_0 = nn.ModuleList()
        self.W_r = nn.ModuleList()
        for i in range(relation_cnt):
            self.W_r.append(nn.ModuleList())

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W_0.append(nn.Linear(input_dim, self.mem_dim))
            for W in self.W_r:
                W.append(nn.Linear(input_dim, self.mem_dim))
        

    def forward(self, nodes, adj):
        gcn_inputs = self.in_drop(nodes)
        maskss = []
        denomss = []
        for batch in range(adj.shape[0]):
            masks = []
            denoms = []
            for i in range(self.relation_cnt):
                if adj[batch, i]._nnz() == 0:
                    continue
                denom = torch.sparse.sum(adj[batch, i], dim=1).to_dense()
                t_g = denom + torch.sparse.sum(adj[batch, i], dim=0).to_dense()
                mask = t_g.eq(0)
                denoms.append(denom.unsqueeze(1))
                masks.append(mask)
            denoms = torch.sum(torch.stack(denoms), 0)
            denoms = denoms + 1
            masks = sum(masks)
            maskss.append(masks)
            denomss.append(denoms)
        denomss = torch.stack(denomss) 

        rgcn_hidden = []
        for l in range(self.layers):
            gAxWs = []
            for j in range(self.relation_cnt):
                gAxW = []

                bxW = self.W_r[j][l](gcn_inputs)
                for batch in range(adj.shape[0]):
                    xW = bxW[batch] 
                    AxW = torch.sparse.mm(adj[batch][j], xW)
                    gAxW.append(AxW)
                gAxW = torch.stack(gAxW)
                gAxWs.append(gAxW)
            gAxWs = torch.stack(gAxWs, dim=1)
            gAxWs = F.relu((torch.sum(gAxWs, 1) + self.W_0[l](gcn_inputs)) / denomss)  # self loop
            gcn_inputs = self.gcn_drop(gAxWs)
            rgcn_hidden.append(gcn_inputs)
        return rgcn_hidden
