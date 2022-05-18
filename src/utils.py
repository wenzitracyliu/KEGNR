import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def adjacency_maxtrix(nodes, q_doc_dict):
    nodes = np.array(nodes)
    xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')
    # print(nodes)
    r_id, c_id = nodes[xv, 0], nodes[yv, 0]
    r_Nid, c_Nid = nodes[xv, 4], nodes[yv, 4]  # node type id
    r_Sid, c_Sid = nodes[xv, 3], nodes[yv, 3]

    adjacency = np.full((r_id.shape[0], r_id.shape[0]), 0.0)
    rgcn_adjacency = np.full((5, r_id.shape[0], r_id.shape[0]), 0.0)

    # query term node-query term node
    adjacency = np.where((r_Nid == 0) & (c_Nid == 0), 1.0, adjacency)
    rgcn_adjacency[0] = np.where((r_Nid == 0) & (c_Nid == 0), 1.0, rgcn_adjacency[0])
    rgcn_adjacency[0] = rgcn_adjacency[0] - np.identity(r_id.shape[0])

    # doc term node-doc term node
    adjacency = np.where((r_Nid == 1) & (c_Nid == 1) & (r_Sid == c_Sid), 1.0, adjacency)
    rgcn_adjacency[1] = np.where((r_Nid == 1) & (c_Nid == 1) & (r_Sid == c_Sid), 1.0, rgcn_adjacency[1])
    rgcn_adjacency[1] = rgcn_adjacency[1] - np.identity(r_id.shape[0])

    # sentence node-sentence node
    adjacency = np.where((r_Nid == 2) & (c_Nid == 2), 1.0, adjacency)
    rgcn_adjacency[2] = np.where((r_Nid == 2) & (c_Nid == 2), 1.0, rgcn_adjacency[2])
    rgcn_adjacency[2] = rgcn_adjacency[2] - np.identity(r_id.shape[0])


    # sentences in order
    # adjacency = np.where((r_Nid == 2) & (c_Nid == 2) & (r_Sid == c_Sid - 1), 1.0, adjacency)
    # adjacency = np.where((r_Nid == 2) & (c_Nid == 2) & (r_Sid - 1 == c_Sid), 1.0, adjacency)
    # rgcn_adjacency[2] = np.where((r_Nid == 2) & (c_Nid == 2) & (r_Sid == c_Sid - 1), 1.0, rgcn_adjacency[2])
    # rgcn_adjacency[2] = np.where((r_Nid == 2) & (c_Nid == 2) & (r_Sid - 1 == c_Sid), 1.0, rgcn_adjacency[2])

    # doc term node-sentence node
    adjacency = np.where((r_Nid == 1) & (c_Nid == 2) & (r_Sid == c_Sid), 1.0, adjacency)
    adjacency = np.where((r_Nid == 2) & (c_Nid == 1) & (r_Sid == c_Sid), 1.0, adjacency)
    rgcn_adjacency[3] = np.where((r_Nid == 1) & (c_Nid == 2) & (r_Sid == c_Sid), 1.0, rgcn_adjacency[3])
    rgcn_adjacency[3] = np.where((r_Nid == 2) & (c_Nid == 1) & (r_Sid == c_Sid), 1.0, rgcn_adjacency[3])

    # query term node-doc term node
    for query_idx, doc_idx in q_doc_dict.items():
        q_idx = int(query_idx)
        if doc_idx:
            r_mat = np.full((r_id.shape[0], r_id.shape[0]), 0) > 1
            c_mat = np.full((r_id.shape[0], r_id.shape[0]), 0) > 1
            for d_idx in doc_idx:
                # d_idx = int(d_idx)
                r_mat = r_mat | (r_id == d_idx)
                c_mat = c_mat | (c_id == d_idx)
            adjacency = np.where((r_id == q_idx) & c_mat, 1.0, adjacency)
            adjacency = np.where(r_mat & (c_id == q_idx), 1.0, adjacency)
            rgcn_adjacency[4] = np.where((r_id == q_idx) & c_mat, 1.0, rgcn_adjacency[4])
            rgcn_adjacency[4] = np.where(r_mat & (c_id == q_idx), 1.0, rgcn_adjacency[4])

        # query term node-sentence node
        # for x, y in zip(xv.ravel(), yv.ravel()):
        #     if nodes[x, 4] == 0 and nodes[y, 4] == 2 and str(nodes[x, 0]) in q_doc_dict:
        #         doc_idx = q_doc_dict[str(nodes[x, 0])]
        #         if doc_idx:
        #             r_mat = np.full((r_id.shape[0], r_id.shape[0]), 0) > 1
        #             for d_idx in doc_idx:
        #                 r_mat = r_mat | (r_id == d_idx)
        #
        #             # at least one doc term in sentence
        #             z = np.where(r_mat & (c_Nid == 2) & (c_Sid == nodes[y, 3]))
        #             temp_ = np.where((r_id == 1) & (c_id == 2) & (r_Sid == c_Sid), 1, adjacency)
        #             temp_ = np.where((r_id == 2) & (c_id == 1) & (r_Sid == c_Sid), 1, temp_)
        #             adjacency[x, y] = 1 if (temp_[z] == 1).any() else 0
        #             adjacency[y, x] = 1 if (temp_[z] == 1).any() else 0
        #             rgcn_adjacency[5][x, y] = 1 if (temp_[z] == 1).any() else 0
        #             rgcn_adjacency[5][y, x] = 1 if (temp_[z] == 1).any() else 0

    rgcn_adjacency[rgcn_adjacency == -1] = 0

    return adjacency, rgcn_adjacency


def convert_3d_to_4d(mxs):
    """
    :param mxs: [3d_tensor]
    :return:
    """
    max_shape = 0
    for mx in mxs:
        max_shape = max(max_shape, mx.shape[1])

    batch_adj = []
    for mx in mxs:
        batch_adj.append(np.pad(mx, ((0, 0), (0, max_shape-mx.shape[1]), (0, max_shape-mx.shape[1])), 'constant'))

    return torch.as_tensor(batch_adj).float()


def convert_2d_to_3d(mxs):
    """
    :param mxs: [3d_tensor]
    :return:
    """
    max_shape = 0
    for mx in mxs:
        max_shape = max(max_shape, mx.shape[0])

    batch_adj = []
    for mx in mxs:
        batch_adj.append(np.pad(mx, ((0, max_shape-mx.shape[0]), (0, max_shape-mx.shape[0])), 'constant', constant_values=1e-8))

    return torch.as_tensor(batch_adj).float()


def split_n_pad(nodes, section, pad=0, return_mask=False):
    """
    split tensor and pad
    :param nodes:
    :param section:
    :param pad:
    :param return_mask:
    :return:
    """
    assert nodes.shape[0] == sum(section.cpu().tolist())
    nodes = torch.split(nodes, section.cpu().tolist())
    nodes = pad_sequence(nodes, batch_first=True, padding_value=pad)
    if not return_mask:
        return nodes
    else:
        max_v = max(section.tolist())
        temp_ = torch.arange(max_v).unsqueeze(0).repeat(nodes.size(0), 1).to(nodes)
        mask = (temp_ < section.unsqueeze(1))

        # mask = torch.zeros(nodes.size(0), max_v).to(nodes)
        # for index, sec in enumerate(section.tolist()):
        #    mask[index, :sec] = 1
        # assert (mask1==mask).all(), print(mask1)
        return nodes, mask


def rm_pad(_input, lens, max_v=None):
    """
    :param _input: batch_size * len * dim
    :param lens: batch_size
    :param max_v:
    :return:
    """
    if max_v is None:
        max_v = lens.max()
    temp_ = torch.arange(max_v).unsqueeze(0).repeat(lens.size(0), 1).to(_input.device)
    remove_pad = (temp_ < lens.unsqueeze(1))
    return _input[remove_pad]


if __name__ == "__main__":
    pass
