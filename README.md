# GCNST

Submission to EMNLP 2020

GCN CCG supertagger

## Prerequisites
* python 3.6
* pytorch 1.1

    
Install python dependencies by running:

`
pip install -r requirements.txt
`

Models are trained on NVIDIA Tesla P100 GPU with 16G RAM

To train a model on a small dataset, run:

`
./run.sh
`

The hyper-parameters for the best performing models are reported in the "./run.sh" file.

You do not need any extra steps if you only want to run and evaluate the supertagging part of our model.

If you want to evaluate the parsing part of our model, you need java1.8 to run "tag2auto.jar".

You also need to setup the C&C parser follow the instruction here: https://aclweb.org/aclwiki/Training_the_C%26C_Parser
