#!/bin/sh

chmod +x auto2dep.sh
mkdir logs
mkdir models
mkdir tmp

CUDA_VISIBLE_DEVICES=2 python supertagging_main.py --do_train --train_data_path=./tmp_data/train.tsv --dev_data_path=./tmp_data/dev.tsv --eval_data_dir=./tmp_data/ --do_lower_case --use_bert --gcn_layer_number=2 --bert_model=/path/to/bert_large_uncased --max_seq_length=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --gradient_accumulation_steps=2 --patient=15 --max_ngram_length=10 --ngram_freq_threshold=2 --model_name=test

