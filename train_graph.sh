#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
        -task classification \
        -model graph \
        -encoder_type bert \
        -train ./data/test_1.jsonl \
        -max_input 1280000 \
        -save ./checkpoints/graph_layer2_dim256_drop0.5_fc3_no_kb_test_2020.bin \
        -dev ./data/trec_graph_dev_no_kb.jsonl \
        -qrels ./data/train_dev_qrels.txt \
        -vocab allenai/scibert_scivocab_uncased \
        -pretrain allenai/scibert_scivocab_uncased \
        -res ./results/graph_test_layer2_dim256_drop0.5_fc3_no_kb_test_2020.trec \
        -metric ndcg_cut_20 \
        -seq_max_len 509 \
        -graph_hidden_dim 256 \
        -graph_layer_nums 2 \
        -relation_cnt 5 \
        -keep_ratio 0.5 \
        -epoch 2 \
        -batch_size 1 \
        -lr 2e-5 \
        -eval_every 50

