#!/bin/sh

chmod +x auto2dep.sh
mkdir logs
mkdir models
mkdir tmp

python nest_ccg_main.py --do_train --train_data_path=./tmp_data/train.tsv --dev_data_path=./tmp_data/dev.tsv --eval_data_dir=./tmp_data/ --do_lower_case --use_bert --use_gcn --use_weight --use_in_chunk --use_cross_chunk --gcn_layer_number=2 --bert_model=/path/to/bert_base_uncased --max_seq_length=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=3 --warmup_proportion=0.1 --learning_rate=1e-5 --patient=15 --max_ngram_length=10 --ngram_freq_threshold=2 --model_name=test
