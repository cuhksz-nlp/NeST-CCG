# NeST-CCG

This is the implementation of [Supertagging Combinatory Categorial Grammar with Attentive Graph Convolutional Networks](https://www.aclweb.org/anthology/) at EMNLP2020.

You can e-mail Yuanhe Tian at `yhtian@uw.edu` or Guimin Chen at `chenguimin@chuangxin.com`, if you have any questions.

## Prerequisites

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`


Install python dependencies by running:

`
pip install -r requirements.txt
`

## Downloading BERT and Our Pre-trained Models

In our paper, we use [BERT](https://www.aclweb.org/anthology/N19-1423/) as the encoder.

For BERT, please download pre-trained BERT model from [Google](https://github.com/google-research/bert) and convert the model from the TensorFlow version to PyTorch version.

For our pre-trained model, we will release it soon.

## Run on Sample Data

To train a model on a small dataset, run:

`
./run.sh
`

## Datasets

We use [CCGbank](https://catalog.ldc.upenn.edu/LDC2005T13) in our paper.

We will release the code to pre-process the data soon.


## Training and Testing

You can find the command lines to train and test models on a specific dataset in `run.sh`.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_test`: test the model.
* `--use_bert`: use BERT as encoder.
* `--use_gcn`: whether to use GCN.
* `--use_weight`: whether to use A-GCN.
* `--use_in_chunk`: whether to in-chunk edges to build the graph.
* `--use_cross_chunk`: whether to cross-chunk edges to build the graph.
* `--gcn_layer_number`: the number of GCN layers.
* `--bert_model`: the directory of pre-trained BERT model.
* `--max_ngram_length`: the max length of n-grams.
* `--ngram_freq_threshold`: n-grams whose frequency is lower than the threshold will be excluded from the lexicon N.
* `--model_name`: the name of model to save.

## Predicting

In processing.


## To-do List

* Release the code to pre-process the data.
* Release the pre-trained model for CCG supertagging.
* Implement the `predict` function of our model.
* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.

You can check our updates at [updates.md](./updates.md).
