import torch
import torch.nn as nn
from src.rgcn import RelationGraph
import torch.nn.functional as F


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, relation_cnt, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.activation = activation
        self.keep_ratio = keep_ratio
        self.attn_gcn = RelationGraph(input_dim, 1, relation_cnt)

    def top_rank(self, attn_score, batch_nodes_nums):
        mask = attn_score.new_zeros((attn_score.shape[0], attn_score.shape[1]), dtype=torch.bool)
        keep_graph_node_num = (self.keep_ratio * batch_nodes_nums).ceil().long()
        sorted_score, sorted_index = attn_score.sort(dim=-1, descending=True)
        keep_node_score = []
        keep_node_index = []

        for i, num in enumerate(keep_graph_node_num):
            mask[i, sorted_index[i, :num]] = True
            keep_node_index.append(sorted_index[i, :num])
            keep_node_score.append(sorted_score[i, :num])

        return mask, keep_graph_node_num, keep_node_index, keep_node_score

    def forward(self, nodes, adjacency, batch_node_nums):
        attn_score = self.attn_gcn(nodes, adjacency).squeeze(-1)
        attn_score = self.activation(attn_score)
        mask, keep_graph_node_num, keep_node_index, keep_node_score = self.top_rank(attn_score, batch_node_nums)
        hidden = nodes[mask] * attn_score[mask].view(-1, 1)

        return hidden, keep_graph_node_num, keep_node_index, keep_node_score


if __name__ == '__main__':
    # rgcn = RelationGraph(2, 1, 2)
    # pool = SelfAttentionPooling(2, 0.5, 2)
    # rgcn.eval()
    # n = torch.FloatTensor([[[1, 1], [2, 2], [2, 2], [2, 2], [0, 0]], [[3, 3], [4, 4], [3, 3], [4, 4], [3, 3]]])
    # adj = torch.randn(2, 2, 5, 5)
    # score = rgcn(n, adj)
    # h, node_num, index = pool(n, adj, torch.LongTensor([4, 5]))
    # print(h)
    # print(node_num)
    pass
