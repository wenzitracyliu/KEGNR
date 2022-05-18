from typing import Tuple

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel
from src.rgcn import RelationGraph
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils import rm_pad, split_n_pad
from torch.nn.utils.rnn import pad_sequence
from src.graph_pooling import SelfAttentionPooling
from src.readout import global_max_pool, global_mean_pool


class GIR(nn.Module):
    def __init__(
        self,
        pretrained: str,
        graph_hidden_dim: int,
        graph_layer_nums: int,
        relation_cnt: int,
        keep_ratio: int,
        task: str = 'ranking'
    ) -> None:
        super(GIR, self).__init__()

        self.layer_nums = graph_layer_nums
        self.config = AutoConfig.from_pretrained(pretrained)
        self.encoder = AutoModel.from_pretrained(pretrained, config=self.config)
        self.graph_layers = nn.ModuleList()

        for layer in range(self.layer_nums):
            input_dim = self.config.hidden_size if layer == 0 else graph_hidden_dim
            self.graph_layers.append(RelationGraph(input_dim, graph_hidden_dim, relation_cnt))
        self.pooling = SelfAttentionPooling(graph_hidden_dim * self.layer_nums, keep_ratio, relation_cnt)

        self.fc1 = nn.Linear(graph_hidden_dim * self.layer_nums * 2, graph_hidden_dim * 2)
        self.fc2 = nn.Linear(graph_hidden_dim * 2, graph_hidden_dim)

        if task == 'ranking':
            self.dense = nn.Linear(graph_hidden_dim, 1)
        elif task == 'classification':
            self.dense = nn.Linear(graph_hidden_dim, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    @staticmethod
    def merge_tokens(info, enc_seq, tp="mean"):
        """
        Merge tokens into mentions;
        Find which tokens belong to a mention (based on start-end ids) and average them
        @:param enc_seq all_word_len * dim  4469*192
        """
        mentions = []
        for i in range(info.shape[0]):
            if tp == "max":
                mention = torch.max(enc_seq[info[i, 1]: info[i, 2], :], dim=-2)[0]
            else:  # mean
                mention = torch.mean(enc_seq[info[i, 1]: info[i, 2], :], dim=-2)
            # print(enc_seq[info[i, 1]: info[i, 2], :])
            # print(info[i, 1], info[i, 2])
            mentions.append(mention)
        if mentions:
            # print('aaaa')
            mentions = torch.stack(mentions)
        return mentions

    def node_layer(self, encoded_seq, q_term_info, doc_term_info, doc_sen_idxes, word_sec):
        # SENTENCE NODES
        sentences = torch.mean(encoded_seq, dim=1)  # sentence nodes (avg of sentence words)
        sentences = torch.index_select(sentences, 0, doc_sen_idxes)  # select doc sentences

        # MENTION & ENTITY NODES
        encoded_seq_token = rm_pad(encoded_seq, word_sec)  # [all_batch_words, word_emb]

        q_terms = self.merge_tokens(q_term_info, encoded_seq_token)
        doc_terms = self.merge_tokens(doc_term_info, encoded_seq_token)  # entity nodes
        # e + m + s (all)
        nodes = q_terms.new_empty((0,), dtype=torch.float)

        nodes = torch.cat((nodes, q_terms), dim=0)
        if type(doc_terms) != list:
            nodes = torch.cat((nodes, doc_terms), dim=0)

        nodes = torch.cat((nodes, sentences), dim=0)
        return nodes

    def graph_layer(self, nodes, section):
        """
        Graph Layer -> Construct a document-level graph
        The graph edges hold representations for the connections between the nodes.
        Args:
            nodes:
            section:     (Tensor <B, 3>) #entities/#mentions/#sentences per batch
        Returns:
        """
        # all nodes in order: entities - mentions - sentences
        # re-order nodes per document (batch)
        nodes = self.rearrange_nodes(nodes, section)

        nodes = split_n_pad(nodes, section.sum(dim=-1))  # torch.Size([4, 76, 210]) batch_size * node_size * node_emb

        return nodes

    @staticmethod
    def rearrange_nodes(nodes, section):
        """
        Re-arrange nodes so that they are in 'Entity - Mention - Sentence' order for each document (batch)
        """
        tmp1 = section.t().contiguous().view(-1).long().to(nodes.device)
        tmp3 = torch.arange(section.numel()).view(section.size(1),
                                                  section.size(0)).t().contiguous().view(-1).long().to(nodes.device)
        tmp2 = torch.arange(section.sum()).to(nodes.device).split(tmp1.tolist())
        tmp2 = pad_sequence(tmp2, batch_first=True, padding_value=-1)[tmp3]
        tmp2 = tmp2[tmp2 != -1]  # remove -1 (padded)

        nodes = torch.index_select(nodes, 0, tmp2)
        return nodes

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor, segment_ids: torch.Tensor,
                token_starts: torch.Tensor, q_term_nodes: torch.Tensor, doc_term_nodes: torch.Tensor,
                section: torch.Tensor, sen_len: torch.Tensor, doc_sen_idxes: torch.Tensor, rgcn_adj: torch.Tensor
                ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        context_output = self.encoder(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                          zip(context_output[0], token_starts)]


        context_output_pad = []
        for output, doc_len in zip(context_output, section[:, 3]):

            if output.size(0) < doc_len:
                padding = Variable(output.detach().new_zeros((1, 1), dtype=torch.float))
                output = torch.cat([output, padding.expand(doc_len - output.size(0), output.size(1))], dim=0)

            context_output_pad.append(output)

        context_output = torch.cat(context_output_pad, dim=0)
        # print(context_output[0])
        encoded_seq = split_n_pad(context_output, sen_len)  # [all_batch_sens, sen_len, word_emb]

        # Graph
        nodes = self.node_layer(encoded_seq, q_term_nodes, doc_term_nodes, doc_sen_idxes, sen_len)

        nodes = self.graph_layer(nodes, section[:, 0:3])

        graph_out = []
        input_nodes = nodes
        for l in range(self.layer_nums):
            input_nodes = F.relu(self.graph_layers[l](input_nodes, rgcn_adj))
            graph_out.append(input_nodes)

        graph_feature = torch.cat(graph_out, dim=-1)

        pool, graph_node_num, keep_node_index, keep_node_score = self.pooling(graph_feature, rgcn_adj, section[:, 0:3].sum(dim=-1))

        readout = torch.cat((global_max_pool(pool, graph_node_num), global_mean_pool(pool, graph_node_num)), dim=-1)

        score = self.dense(F.relu(self.fc2(F.relu(self.fc1(readout))))).squeeze(-1)

        return score, keep_node_index, keep_node_score
