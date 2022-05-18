#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 \
python train.py \
        -task classification \
        -model bert \
        -train ./data/trec_clas_2_train.jsonl \
        -max_input 1280000 \
        -save ./checkpoints/pubmedbert_ranking_knowledge.bin \
        -dev ./data/trec_knowledge_dev.jsonl \
        -qrels ./data/train_dev_qrels.txt \
        -vocab allenai/scibert_scivocab_uncased \
        -pretrain allenai/scibert_scivocab_uncased \
        -res ./results/scibert_dev_ranking_knowledge.trec \
        -metric ndcg_cut_20 \
        -max_query_len 50 \
        -max_doc_len 450 \
        -epoch 2 \
        -batch_size 4 \
        -lr 2e-5 \
        -eval_every 50
