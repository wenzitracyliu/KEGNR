import os
import json
from argparse import Action
import random
import numpy
import torch


class DictOrStr(Action):
    def __call__(self, parser, namespace, values, option_string=None):
         if '=' in values:
             my_dict = {}
             for kv in values.split(","):
                 k,v = kv.split("=")
                 my_dict[k] = v
             setattr(namespace, self.dest, my_dict)
         else:
             setattr(namespace, self.dest, values)


def set_environment(seed, set_cuda=False):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(q_id+' Q0 '+str(value[0])+' '+str(rank+1)+' '+str(value[1][0])+' openmatch\n')
    return


def save_features(rst_file, features):
    with open(rst_file, 'w') as writer:
        for feature in features:
            writer.write(feature+'\n')
    return


def save_index(node_file, index, score):
    with open(node_file, 'w') as writer:
        for q_id, d_index in index.items():
            for doc_id, idx in d_index.items():
                s = score[q_id][doc_id]
                writer.write(
                    q_id + ' ' + str(doc_id) + ' ' + '/'.join([str(d_idx) for d_idx in idx]) + '\t\t' +
                    '/'.join([str(round(d_s, 6)) for d_s in s]) + '\n')
    return
