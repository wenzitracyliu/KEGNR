#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task ranking \
        -model tk \
        -max_input 1280000 \
        -vocab ./data/glove.6B.300d.txt \
        -checkpoint ./checkpoints/tk_knowledge_ranking_2.bin \
        -test ./data/trec_knowledge_test_2.jsonl \
        -res ./results/tk_test_knowledge_ranking_2.trec \
        -n_kernels 21 \
        -max_query_len 32 \
        -max_doc_len 477 \
        -batch_size 32
