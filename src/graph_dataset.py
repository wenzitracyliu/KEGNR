from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
from src.utils import convert_3d_to_4d, adjacency_maxtrix
from transformers import AutoTokenizer
from src.transformers_word_handle import transformers_word_handle


class GraphDataset(Dataset):
    def __init__(
            self,
            dataset: str,
            tokenizer: AutoTokenizer,
            mode: str,
            encoder_type: str = 'bert',
            seq_max_len: int = 16,
            max_input: int = 1280000,
            task: str = 'ranking'
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._mode = mode
        self._max_input = max_input
        self._task = task
        if seq_max_len + 3 > 512:
            raise ValueError('seq_max_len + 3 > 512.')
        self.input_handle = transformers_word_handle(encoder_type, tokenizer, seq_max_len)

        if isinstance(self._dataset, str):

            with open(self._dataset, 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
                    line = json.loads(line)
                    self._examples.append(line)
        else:
            raise ValueError('Data path must be `str`.')
        self._count = len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        if self._mode == 'train':
            if self._task == 'ranking':
                input_ids_pos = torch.LongTensor([item['input_ids_pos'] for item in batch])
                segment_ids_pos = torch.LongTensor([item['segment_ids_pos'] for item in batch])
                input_mask_pos = torch.LongTensor([item['input_mask_pos'] for item in batch])
                token_starts_pos = torch.BoolTensor([item['token_starts_pos'] for item in batch])
                section_pos = torch.LongTensor([item['section_pos'] for item in batch])

                input_ids_neg = torch.LongTensor([item['input_ids_neg'] for item in batch])
                segment_ids_neg = torch.LongTensor([item['segment_ids_neg'] for item in batch])
                input_mask_neg = torch.LongTensor([item['input_mask_neg'] for item in batch])
                token_starts_neg = torch.BoolTensor([item['token_starts_neg'] for item in batch])
                section_neg = torch.LongTensor([item['section_neg'] for item in batch])

                batch_query_term_pos = []
                batch_doc_term_pos = []
                word_count_pos, sen_count_pos = 0, 0
                batch_doc_sen_idx_pos = []
                batch_sen_len_pos = []

                batch_query_term_neg = []
                batch_doc_term_neg = []
                word_count_neg, sen_count_neg = 0, 0
                batch_doc_sen_idx_neg = []
                batch_sen_len_neg = []
                for item in batch:
                    for q in item['q_term_nodes_pos']:
                        batch_query_term_pos.append([q[0], q[1] + word_count_pos, q[2] + word_count_pos, q[3] + sen_count_pos, q[4]])

                    for d in item['doc_term_nodes_pos']:
                        batch_doc_term_pos.append([d[0], d[1] + word_count_pos, d[2] + word_count_pos, d[3] + sen_count_pos, d[4]])

                    for q in item['q_term_nodes_neg']:
                        batch_query_term_neg.append([q[0], q[1] + word_count_neg, q[2] + word_count_neg, q[3] + sen_count_neg, q[4]])

                    for d in item['doc_term_nodes_neg']:
                        batch_doc_term_neg.append([d[0], d[1] + word_count_neg, d[2] + word_count_neg, d[3] + sen_count_neg, d[4]])

                    doc_sen_idx_pos = [i + sen_count_pos for i in range(len(item['sen_len_pos']))]
                    doc_sen_idx_neg = [i + sen_count_neg for i in range(len(item['sen_len_neg']))]

                    word_count_pos += item['section_pos'][3]
                    sen_count_pos += len(item['sen_len_pos'])
                    batch_doc_sen_idx_pos.extend(doc_sen_idx_pos)
                    batch_sen_len_pos.extend(item['sen_len_pos'])

                    word_count_neg += item['section_neg'][3]
                    sen_count_neg += len(item['sen_len_neg'])
                    batch_doc_sen_idx_neg.extend(doc_sen_idx_neg)
                    batch_sen_len_neg.extend(item['sen_len_neg'])

                q_term_nodes_pos = torch.LongTensor(batch_query_term_pos)
                doc_term_nodes_pos = torch.LongTensor(batch_doc_term_pos)
                doc_sen_idxes_pos = torch.LongTensor(batch_doc_sen_idx_pos)
                sen_len_pos = torch.LongTensor(batch_sen_len_pos)

                rgcn_adjacency_pos = convert_3d_to_4d([item['rgcn_adjacency_pos'] for item in batch])

                q_term_nodes_neg = torch.LongTensor(batch_query_term_neg)
                doc_term_nodes_neg = torch.LongTensor(batch_doc_term_neg)
                doc_sen_idxes_neg = torch.LongTensor(batch_doc_sen_idx_neg)
                sen_len_neg = torch.LongTensor(batch_sen_len_neg)

                rgcn_adjacency_neg = convert_3d_to_4d([item['rgcn_adjacency_neg'] for item in batch])

                return {'input_ids_pos': input_ids_pos, 'segment_ids_pos': segment_ids_pos,
                        'input_mask_pos': input_mask_pos, 'token_starts_pos': token_starts_pos,
                        'q_term_nodes_pos': q_term_nodes_pos, 'doc_term_nodes_pos': doc_term_nodes_pos, 'section_pos': section_pos,
                        'sen_len_pos': sen_len_pos, 'doc_sen_idxes_pos': doc_sen_idxes_pos, 'rgcn_adjacency_pos': rgcn_adjacency_pos,
                        'input_ids_neg': input_ids_neg, 'segment_ids_neg': segment_ids_neg,
                        'input_mask_neg': input_mask_neg, 'token_starts_neg': token_starts_neg,
                        'q_term_nodes_neg': q_term_nodes_neg, 'doc_term_nodes_neg': doc_term_nodes_neg, 'section_neg': section_neg,
                        'sen_len_neg': sen_len_neg, 'doc_sen_idxes_neg': doc_sen_idxes_neg, 'rgcn_adjacency_neg': rgcn_adjacency_neg
                        }

            elif self._task == 'classification':

                input_ids = torch.LongTensor([item['input_ids'] for item in batch])
                segment_ids = torch.LongTensor([item['segment_ids'] for item in batch])
                input_mask = torch.LongTensor([item['input_mask'] for item in batch])
                label = torch.LongTensor([item['label'] for item in batch])
                token_starts = torch.BoolTensor([item['token_starts'] for item in batch])
                section = torch.LongTensor([item['section'] for item in batch])

                batch_query_term = []
                batch_doc_term = []
                word_count, sen_count = 0, 0
                batch_doc_sen_idx = []
                batch_sen_len = []
                for item in batch:
                    for q in item['q_term_nodes']:
                        batch_query_term.append([q[0], q[1] + word_count, q[2] + word_count, q[3] + sen_count, q[4]])

                    for d in item['doc_term_nodes']:
                        batch_doc_term.append([d[0], d[1] + word_count, d[2] + word_count, d[3] + sen_count, d[4]])

                    doc_sen_idx = [i + sen_count for i in range(len(item['sen_len']))]
                    word_count += item['section'][3]
                    sen_count += len(item['sen_len'])
                    batch_doc_sen_idx.extend(doc_sen_idx[1:])
                    batch_sen_len.extend(item['sen_len'])

                q_term_nodes = torch.LongTensor(batch_query_term)
                doc_term_nodes = torch.LongTensor(batch_doc_term)
                doc_sen_idxes = torch.LongTensor(batch_doc_sen_idx)
                sen_len = torch.LongTensor(batch_sen_len)

                rgcn_adjacency = convert_3d_to_4d([item['rgcn_adjacency'] for item in batch])

                return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask,
                        'token_starts': token_starts, 'label': label,
                        'q_term_nodes': q_term_nodes, 'doc_term_nodes': doc_term_nodes, 'section': section,
                        'sen_len': sen_len, 'doc_sen_idxes': doc_sen_idxes, 'rgcn_adjacency': rgcn_adjacency}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev' or self._mode == 'test':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]

            input_ids = torch.LongTensor([item['input_ids'] for item in batch])
            segment_ids = torch.LongTensor([item['segment_ids'] for item in batch])
            input_mask = torch.LongTensor([item['input_mask'] for item in batch])
            label = torch.LongTensor([item['label'] for item in batch])
            token_starts = torch.BoolTensor([item['token_starts'] for item in batch])
            section = torch.LongTensor([item['section'] for item in batch])

            batch_query_term = []
            batch_doc_term = []
            word_count, sen_count = 0, 0
            batch_doc_sen_idx = []
            batch_sen_len = []
            for item in batch:
                for q in item['q_term_nodes']:
                    batch_query_term.append([q[0], q[1] + word_count, q[2] + word_count, q[3] + sen_count, q[4]])

                for d in item['doc_term_nodes']:
                    batch_doc_term.append([d[0], d[1] + word_count, d[2] + word_count, d[3] + sen_count, d[4]])

                doc_sen_idx = [i + sen_count for i in range(len(item['sen_len']))]
                word_count += item['section'][3]
                sen_count += len(item['sen_len'])
                batch_doc_sen_idx.extend(doc_sen_idx[1:])
                batch_sen_len.extend(item['sen_len'])

            q_term_nodes = torch.LongTensor(batch_query_term)
            doc_term_nodes = torch.LongTensor(batch_doc_term)
            doc_sen_idxes = torch.LongTensor(batch_doc_sen_idx)
            sen_len = torch.LongTensor(batch_sen_len)

            rgcn_adjacency = convert_3d_to_4d([item['rgcn_adjacency'] for item in batch])

            return {'query_id': query_id, 'doc_id': doc_id,
                    'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask,
                    'token_starts': token_starts, 'label': label,
                    'q_term_nodes': q_term_nodes, 'doc_term_nodes': doc_term_nodes, 'section': section,
                    'sen_len': sen_len, 'doc_sen_idxes': doc_sen_idxes, 'rgcn_adjacency': rgcn_adjacency}

        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]

        if self._mode == 'train':
            if self._task == 'ranking':
                input_ids_pos, input_mask_pos, segment_ids_pos, token_starts_pos = \
                    self.input_handle.subword_tokenize_to_ids(example['query'], example['doc_pos'])

                input_ids_neg, input_mask_neg, segment_ids_neg, token_starts_neg = \
                    self.input_handle.subword_tokenize_to_ids(example['query'], example['doc_neg'])

                _, rgcn_adjacency_pos = adjacency_maxtrix(example['q_term_nodes_pos'] + example['doc_term_nodes_pos']
                                                          + example['sen_nodes_pos'], example['q_doc_dict_pos'])

                _, rgcn_adjacency_neg = adjacency_maxtrix(example['q_term_nodes_neg'] + example['doc_term_nodes_neg']
                                                          + example['sen_nodes_neg'], example['q_doc_dict_neg'])

                # -1 means remove query sentence
                return {'input_ids_pos': input_ids_pos, 'segment_ids_pos': segment_ids_pos,
                        'input_mask_pos': input_mask_pos, 'token_starts_pos': token_starts_pos,
                        'q_term_nodes_pos': example['q_term_nodes_pos'],
                        'doc_term_nodes_pos': example['doc_term_nodes_pos'],
                        'section_pos': [len(example['q_term_nodes_pos']), len(example['doc_term_nodes_pos']),
                                        len(example['sen_len_pos'])-1, sum(example['sen_len_pos'])],
                        'sen_len_pos': example['sen_len_pos'], 'rgcn_adjacency_pos': rgcn_adjacency_pos,

                        'input_ids_neg': input_ids_neg, 'segment_ids_neg': segment_ids_neg,
                        'input_mask_neg': input_mask_neg, 'token_starts_neg': token_starts_neg,
                        'q_term_nodes_neg': example['q_term_nodes_neg'],
                        'doc_term_nodes_neg': example['doc_term_nodes_neg'],
                        'section_neg': [len(example['q_term_nodes_neg']), len(example['doc_term_nodes_neg']),
                                        len(example['sen_len_neg'])-1, sum(example['sen_len_neg'])],
                        'sen_len_neg': example['sen_len_neg'], 'rgcn_adjacency_neg': rgcn_adjacency_neg
                        }
            elif self._task == 'classification':

                input_ids, input_mask, segment_ids, token_starts = \
                    self.input_handle.subword_tokenize_to_ids(example['query'], example['doc'])

                _, rgcn_adjacency = adjacency_maxtrix(example['q_term_nodes'] + example['doc_term_nodes']
                                                      + example['sen_nodes'], example['q_doc_dict'])
                # -1 means remove query sentence
                return {'input_ids': input_ids, 'segment_ids': segment_ids,
                        'input_mask': input_mask, 'token_starts': token_starts,
                        'q_term_nodes': example['q_term_nodes'],
                        'doc_term_nodes': example['doc_term_nodes'],
                        'section': [len(example['q_term_nodes']), len(example['doc_term_nodes']),
                                    len(example['sen_len'])-1, sum(example['sen_len'])],
                        'sen_len': example['sen_len'], 'rgcn_adjacency': rgcn_adjacency,
                        'label': example['label']}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev' or self._mode == 'test':
            input_ids, input_mask, segment_ids, token_starts = \
                self.input_handle.subword_tokenize_to_ids(example['query'], example['doc'])

            _, rgcn_adjacency = adjacency_maxtrix(example['q_term_nodes'] + example['doc_term_nodes']
                                                  + example['sen_nodes'], example['q_doc_dict'])

            # -1 means remove query sentence
            return {'input_ids': input_ids, 'segment_ids': segment_ids,
                    'input_mask': input_mask, 'token_starts': token_starts,
                    'q_term_nodes': example['q_term_nodes'],
                    'doc_term_nodes': example['doc_term_nodes'],
                    'section': [len(example['q_term_nodes']), len(example['doc_term_nodes']),
                                len(example['sen_len'])-1, sum(example['sen_len'])],
                    'sen_len': example['sen_len'], 'rgcn_adjacency': rgcn_adjacency,
                    'label': example['label'], 'query_id': example['query_id'], 'doc_id': example['doc_id']}

        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return self._count
