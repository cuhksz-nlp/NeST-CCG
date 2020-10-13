# NeST-CCG

This is the implementation of [Supertagging Combinatory Categorial Grammar with Attentive Graph Convolutional Networks](https://www.aclweb.org/anthology/) at EMNLP2020.

You can e-mail Yuanhe Tian at `yhtian@uw.edu` or Guimin Chen at `cuhksz.nlp@gmail.com`, if you have any questions.


## Citation

If you use or extend our work, please cite our paper at EMNLP-2020.

```
@inproceedings{tian-etal-2020-supertagging,
    title = "Supertagging Combinatory Categorial Grammar with Attentive Graph Convolutional Networks",
    author = "Tian, Yuanhe and Song, Yan and Xia, Fei",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
}
```

## Prerequisites

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`


Install python dependencies by running:

`
pip install -r requirements.txt
`

You also need `Java 1.8` to run `tag2auto.jar`, which wiil generate the CCG parsing results from the predicted supertags. You can skip this step if you only want to get the supertagging results.

## Evaluation

To evaluate the CCG parsing results generated from the predicted supertags, you need to [setup C&C parser](https://aclweb.org/aclwiki/Training_the_C%26C_Parser). To do this, just run `./setup_condc.sh`.

To test whether C&C parser is successfully installed for the purpose of evaluation, run `./candc/bin/generate`. If you see the following, then it means C&C parser is successfully installed.

```angular2
$ ./candc/bin/generate
expected a flag indicating output type
usage: generate -t <cats_directory> <markedup_file> <pipe_file> > <deps> 2> <feats>
       generate -e <cats_directory> <markedup_file> <pipe_file> > <word_deps>
       generate -j <cats_directory> <markedup_file> <pipe_file> > <word_deps>
       generate -g <cats_directory> <markedup_file> <pipe_file> > <grs>
       generate -T <cats_directory> <markedup_file> <pipe_file> > <raw text>
```

## Downloading BERT and Our Pre-trained Models

In our paper, we use [BERT](https://www.aclweb.org/anthology/N19-1423/) as the encoder.

For BERT, please download pre-trained BERT model from [Google](https://github.com/google-research/bert) and convert the model from the TensorFlow version to PyTorch version.

For our pre-trained model, we you can download it from [Baidu Wangpan](https://pan.baidu.com/s/1YVvUvRPU-wgquwydlhAK0A) (passcode: u4ta) or from Google Drive (coming soon).

## Run on Sample Data

To train a model on [a small dataset](./sample_data), see the command lines in `run.sh`.


## Datasets

We use [CCGbank](https://catalog.ldc.upenn.edu/LDC2005T13) in our paper. 

To preprocess the data, please go to `data_processing` directory and run `./data_processing`. You can find more details [here](./data_processing/README.md). You need to obtain the official CCGbank yourself before running our code.

If everything goes smoothly, you will see all data files in the `./data` directory with all filenames identical with the ones in `./sample_data`.


## Training and Testing

You can find the command lines to train and test models on the sample data in `run.sh`.

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

* Release the pre-trained model for CCG supertagging (on Google Drive).
* Implement the `predict` function of our model.
* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.

You can check our updates at [updates.md](./updates.md).
