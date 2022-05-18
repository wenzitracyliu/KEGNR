from transformers import AutoTokenizer
import numpy as np


class transformers_word_handle():

    def __init__(self, model_type: str, tokenizer: AutoTokenizer, seq_max_len: int):
        super().__init__()
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_len = seq_max_len

    def convert_tokens_to_ids(self, tokens, query_len, doc_len):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.model_type == 'xlnet':

            segment_ids = [0] * (query_len + 1) + [1] * (doc_len + 1) + [2]
        else:
            segment_ids = [0] * (query_len + 2) + [1] * (doc_len + 1)
        token_mask = [1] * len(token_ids)

        padding_len = self.max_len - len(token_ids) + 3
        token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_len
        token_mask = token_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len

        assert len(token_ids) == self.max_len + 3
        assert len(token_mask) == self.max_len + 3
        assert len(segment_ids) == self.max_len + 3

        return token_ids, token_mask, segment_ids

    def flatten(self, list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, query, doc):
        """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        query,doc: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """

        query_subwords = list(map(self.tokenizer.tokenize, query))
        query_subword_lengths = list(map(len, query_subwords))
        query_subwords = list(self.flatten(query_subwords))

        doc_subwords = list(map(self.tokenizer.tokenize, doc))
        doc_subword_lengths = list(map(len, doc_subwords))
        doc_subwords = list(self.flatten(doc_subwords))

        if self.model_type == 'xlnet':  # xlnet 系列
            subwords = query_subwords + [self.tokenizer.sep_token] + doc_subwords[:self.max_len-len(query_subwords)] \
                       + [self.tokenizer.sep_token] + [self.tokenizer.cls_token]
            query_start_idxs = np.cumsum([0] + query_subword_lengths[:-1])
            doc_start_idxs = 1 + len(query_subwords) + np.cumsum([0] + doc_subword_lengths[:-1])
            token_start_idxs = np.concatenate((query_start_idxs, doc_start_idxs), axis=0)
            token_start_idxs[token_start_idxs > self.max_len+1] = self.max_len + 1

        elif self.model_type == 'bert':  # bert
            subwords = [self.tokenizer.cls_token] + query_subwords + [self.tokenizer.sep_token] \
                       + doc_subwords[:self.max_len-len(query_subwords)] + [self.tokenizer.sep_token]
            query_start_idxs = 1 + np.cumsum([0] + query_subword_lengths[:-1])
            doc_start_idxs = 2 + len(query_subwords) + np.cumsum([0] + doc_subword_lengths[:-1])
            token_start_idxs = np.concatenate((query_start_idxs, doc_start_idxs), axis=0)
            token_start_idxs[token_start_idxs > self.max_len+2] = self.max_len + 2
        else:
            raise ValueError('Model_type must be `bert` or `xlnet`.')

        return subwords, token_start_idxs, len(query_subwords), len(doc_subwords[:self.max_len-len(query_subwords)])

    def subword_tokenize_to_ids(self, query, doc):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.
        Parameters
        ----------
        query, doc: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        subwords, token_start_idxs, query_len, doc_len = self.subword_tokenize(query, doc)
        subword_ids, mask, segment_ids = self.convert_tokens_to_ids(subwords, query_len, doc_len)
        token_starts = np.zeros((self.max_len+3))
        token_starts[token_start_idxs] = 1

        return subword_ids, mask, segment_ids, token_starts.tolist()

