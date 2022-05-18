#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
        -task ranking \
        -model knrm \
        -train ./data/trec_knowledge_sample_ranking_train.jsonl \
        -max_input 1280000 \
        -save ./checkpoints/knrm_knowledge_ranking_2.bin \
        -dev ./data/trec_knowledge_dev_2.jsonl \
        -qrels ./data/train_dev_qrels.txt \
        -vocab ./data/glove.6B.300d.txt \
        -res ./results/knrm_dev_knowdege_ranking_2.trec \
        -metric ndcg_cut_20 \
        -n_kernels 31 \
        -max_query_len 32 \
        -max_doc_len 477 \
        -epoch 10 \
        -batch_size 32 \
        -lr 1e-3 \
        -eval_every 50
