import torch
from src.utils import split_n_pad


def global_max_pool(nodes, graph_node_num):
    nodes_pad = split_n_pad(nodes, graph_node_num)
    return torch.max(nodes_pad, dim=1)[0]


def global_mean_pool(nodes, graph_node_num):
    nodes_pad = split_n_pad(nodes, graph_node_num)
    return torch.mean(nodes_pad, dim=1)
