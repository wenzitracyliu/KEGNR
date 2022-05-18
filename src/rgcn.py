import torch
from torch import nn
from torch.nn.parameter import Parameter


class RelationGraph(nn.Module):

    def __init__(self, input_dim, hidden_dim, relation_cnt, dropout=0.5):
        super(RelationGraph, self).__init__()
        self.w_0 = Parameter(torch.FloatTensor(hidden_dim, input_dim))
        self.w_r = Parameter(torch.FloatTensor(relation_cnt, hidden_dim, input_dim))
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w_0)
        nn.init.kaiming_normal_(self.w_r)

    def forward(self, nodes, adj):
        """
        :param nodes:  batch_size * node_size * node_emb
        :param adj:  batch_size * 5 * node_size * node_size
        :return:
        """

        xw_0 = torch.einsum('ijk,lk->ijl', [nodes, self.w_0])
        xw_r = torch.einsum('ijk,mlk->imjl', [nodes, self.w_r])
        assert xw_r.shape == (nodes.shape[0], self.w_r.shape[0], nodes.shape[1], self.w_r.shape[1])

        c_r = torch.sum(adj, axis=-1, keepdim=True)
        c_r[c_r == 0] = 1

        output = torch.sum(xw_r / c_r, axis=1) + xw_0

        output = self.drop(output)

        return output
