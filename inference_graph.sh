#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task classification \
        -model graph \
        -encoder_type bert \
        -max_input 1280000 \
        -test ./data/trec_graph_test.jsonl \
        -vocab allenai/scibert_scivocab_uncased \
        -pretrain allenai/scibert_scivocab_uncased \
        -checkpoint ./checkpoints/graph_layer2_dim256_drop0.5_fc3_test.bin \
        -res ./results/graph_test_layer2_dim256_drop0.5_fc3_test.trec \
        -node_index_path ./results/node_index_test_layer2_dim256_drop0.5_fc3_test.tsv \
        -seq_max_len 509 \
        -graph_hidden_dim 256 \
        -graph_layer_nums 2 \
        -relation_cnt 5 \
        -keep_ratio 0.5 \
        -batch_size 4

#python inference.py \
#        -task classification \
#        -model graph \
#        -encoder_type bert \
#        -max_input 1280000 \
#        -test ./data/trec_graph_test_no_kb.jsonl \
#        -vocab allenai/scibert_scivocab_uncased \
#        -pretrain allenai/scibert_scivocab_uncased \
#        -checkpoint ./checkpoints/graph_layer2_dim256_drop0.5_fc3_no_kb_test.bin \
#        -res ./results/graph_test_layer2_dim256_drop0.5_fc3_no_kb_test.trec \
#        -node_index_path ./results/node_index_test_layer2_dim256_drop0.5_fc3_no_kb_test.tsv \
#        -seq_max_len 509 \
#        -graph_hidden_dim 256 \
#        -graph_layer_nums 2 \
#        -relation_cnt 5 \
#        -keep_ratio 0.5 \
#        -batch_size 4


