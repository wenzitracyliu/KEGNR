#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 \
python inference.py \
        -task classification \
        -model bert \
        -max_input 1280000 \
        -test ./data/trec_knowledge_test_2020.jsonl \
        -vocab allenai/scibert_scivocab_uncased \
        -pretrain allenai/scibert_scivocab_uncased \
        -checkpoint ./checkpoints/scibert_knowledge_2.bin \
        -res ./results/scibert_test_knowledge_2020.trec \
        -max_query_len 10 \
        -max_doc_len 499 \
        -batch_size 4