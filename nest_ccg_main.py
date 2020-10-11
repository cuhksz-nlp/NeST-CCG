from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

from os import path

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from tqdm import tqdm, trange
import subprocess

from nest_ccg_helper import get_vocab, get_labels
from nest_ccg_eval import Evaluation, ccgparse, candc_path
from nest_ccg_model import NeSTCCG
import datetime


def train(args):

    if args.use_bert and args.use_xlnet:
        raise ValueError('We cannot use both BERT and XLNet')

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_name = './logs/log-' + now_time
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_file_name,
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    logger.info(vars(args))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.model_name is None:
        raise Warning('model name is not specified, the model will NOT be saved!')
    output_model_dir = os.path.join('./models', args.model_name + '_' + now_time)

    ngram2id, ngram2count = get_vocab(args.train_data_path,
                                      args.max_ngram_length, args.ngram_freq_threshold)

    label_map = get_labels(args.train_data_path)

    hpara = NeSTCCG.init_hyper_parameters(args)
    supertagger = NeSTCCG(labelmap=label_map, hpara=hpara, model_path=args.bert_model,
                          gram2id=ngram2id)

    train_examples = supertagger.load_data(args.train_data_path, flag='train')
    dev_examples = supertagger.load_data(args.dev_data_path, flag='dev')
    test_examples = supertagger.load_data(args.test_data_path, flag='test')

    all_eval_examples = {'dev': dev_examples, 'test': test_examples}
    num_labels = supertagger.num_labels
    convert_examples_to_features = supertagger.convert_examples_to_features
    clipping_top_n = supertagger.clipping_top_n
    clipping_threshold = supertagger.clipping_threshold
    id2label = supertagger.id2label
    feature2input = supertagger.feature2input

    total_params = sum(p.numel() for p in supertagger.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %d' % total_params)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.fp16:
        supertagger.half()
    supertagger.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        supertagger = DDP(supertagger)
    elif n_gpu > 1:
        supertagger = torch.nn.DataParallel(supertagger)

    param_optimizer = list(supertagger.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    best_eval = -1
    best_info_str = ''
    history = {'epoch': [], 'dev_acc': [], 'dev_cats': [], 'dev_lf': [], 'dev_uf': [],
               'test_acc': [], 'test_cats': [], 'test_lf': [], 'test_uf': []}
    num_of_no_improvement = 0
    patient = args.patient

    evaluator = Evaluation(args.eval_data_dir)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            supertagger.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
                supertagger.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                train_features = convert_examples_to_features(batch_examples)

                input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, \
                dep_adjacency_matrix = feature2input(device, train_features)

                loss, _ = supertagger(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask,
                                      adjacency_matrix=dep_adjacency_matrix)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            supertagger.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                supertagger.eval()

                output_model_dir = path.join('./models', args.model_name + '_' + now_time)
                if not os.path.exists(output_model_dir):
                    os.mkdir(output_model_dir)

                history['epoch'].append(epoch)
                for flag in ['dev', 'test']:
                    eval_examples = all_eval_examples[flag]
                    all_y_true = []
                    all_y_pred = []
                    output_suppertag_list = []
                    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
                        eval_batch_examples = eval_examples[start_index:
                                                            min(start_index + args.eval_batch_size, len(eval_examples))]
                        eval_features = convert_examples_to_features(eval_batch_examples)

                        input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, \
                        dep_adjacency_matrix = feature2input(device, eval_features)

                        with torch.no_grad():
                            _, logits = supertagger(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask,
                                                    adjacency_matrix=dep_adjacency_matrix
                                                    )

                        logits = F.softmax(logits, dim=2)
                        argmax_logits = torch.argmax(logits, dim=2)
                        argsort_loagits = torch.argsort(logits, dim=2, descending=True)
                        argmax_logits = argmax_logits.detach().cpu().numpy()
                        argsort_loagits = argsort_loagits.detach().cpu().numpy()[:, :, : clipping_top_n]
                        logits = logits.to('cpu').numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        input_mask = input_mask.to('cpu').numpy()

                        for i, label in enumerate(label_ids):
                            temp_1 = []
                            temp_2 = []
                            for j, m in enumerate(label):
                                if j == 0:
                                    continue
                                elif label_ids[i][j] == num_labels - 1:
                                    all_y_true.append(temp_1)
                                    all_y_pred.append(temp_2)
                                    break
                                else:
                                    temp_1.append(id2label[label_ids[i][j]])
                                    temp_2.append(id2label[argmax_logits[i][j]])

                        for i in range(len(label_ids)):
                            ex = eval_batch_examples[i]
                            label = label_ids[i]
                            text = ex.text_a.split(' ')
                            output_line = []
                            for j, m in enumerate(label):
                                if j == 0:
                                    continue
                                elif label_ids[i][j] == num_labels - 1:
                                    assert len(text) == j - 1
                                    output_suppertag_list.append('#word#'.join(output_line))
                                    break
                                else:
                                    super_tag_str_list = []
                                    prob_str_list = []
                                    for tag_id in argsort_loagits[i][j]:
                                        if tag_id == 0:
                                            continue
                                        label = id2label[tag_id]
                                        prob = logits[i][j][tag_id]
                                        if len(super_tag_str_list) > 0 and prob < clipping_threshold:
                                            break
                                        else:
                                            super_tag_str_list.append(label)
                                            prob_str_list.append(str(prob))
                                    word_str = text[j - 1] + '\t' + '#'.join(
                                        super_tag_str_list) + '\t' + '#'.join(prob_str_list)
                                    output_line.append(word_str)

                    y_true_all = []
                    y_pred_all = []
                    eval_sentence_all = []
                    for y_true_item in all_y_true:
                        y_true_all += y_true_item
                    for y_pred_item in all_y_pred:
                        y_pred_all += y_pred_item
                    for example, y_true_item in zip(eval_examples, all_y_true):
                        sen = example.text_a
                        sen = sen.strip()
                        sen = sen.split(' ')
                        if len(y_true_item) != len(sen):
                            # print(len(sen))
                            sen = sen[:len(y_true_item)]
                        eval_sentence_all.append(sen)
                    acc = evaluator.supertag_acc(y_pred_all, y_true_all)

                    history[flag + '_acc'].append(acc)

                    auto_output_file = os.path.join(output_model_dir, flag + '.auto')

                    supertag_output_file = os.path.join(output_model_dir, flag + '.supertag.txt')

                    with open(supertag_output_file, 'w', encoding='utf8') as f:
                        for line in output_suppertag_list:
                            f.write(line + '\n')

                    command = 'java -jar ' + ccgparse + ' -f ' + supertag_output_file + ' -o ' + auto_output_file + ' >' + auto_output_file
                    subprocess.run(command, shell=True)

                    dep_output_file = os.path.join(output_model_dir, flag + '.dep')

                    command = './auto2dep.sh ' + candc_path + ' ' + auto_output_file + ' ' + dep_output_file
                    subprocess.run(command, shell=True)

                    eval_output_file = os.path.join(output_model_dir, flag + '.eval')
                    tag_gold = os.path.join(args.eval_data_dir, 'gold_files', flag + '.stagged')
                    dep_gold = os.path.join(args.eval_data_dir, 'gold_files', flag + '.dep.gold')
                    command = 'python ccg_eval.py -r ' + tag_gold + ' ' + dep_gold + ' ' \
                              + dep_output_file + ' ' + auto_output_file + ' >' + eval_output_file
                    subprocess.run(command, shell=True)

                    results = evaluator.eval_file_reader(eval_output_file)

                    for key, value in results.items():
                        h_key = flag + '_' + key
                        if h_key in history:
                            history[h_key].append(value)

                log_info = []
                for key, value in history.items():
                    log_info.append(key)
                    log_info.append(str(value[-1]))
                info_str = ' '.join(log_info)
                logger.info(info_str)

                keep
                if history['dev_acc'][-1] > best_eval:
                    best_eval = history['dev_acc'][-1]
                    best_info_str = info_str
                    num_of_no_improvement = 0

                    model_to_save = supertagger.module if hasattr(supertagger, 'module') else supertagger
                    best_eval_model_dir = os.path.join(output_model_dir, 'model')
                    if not os.path.exists(best_eval_model_dir):
                        os.mkdir(best_eval_model_dir)
                    model_to_save.save_model(best_eval_model_dir, args.bert_model)
                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        logger.info("\n======= best ========\n")
        logger.info(best_info_str)
        logger.info("\n======= best ========\n")

        with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
            json.dump(history, f)
            f.write('\n')


def test(args):

    evaluator = Evaluation(args.eval_data_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    supertagger = NeSTCCG.load_model(args.eval_model)

    eval_examples = supertagger.load_data(args.eval_data_path, flag='test')
    num_labels = supertagger.num_labels
    convert_examples_to_features = supertagger.convert_examples_to_features
    clipping_threshold = supertagger.clipping_threshold
    clipping_top_n = supertagger.clipping_top_n
    id2label = supertagger.id2label
    feature2input = supertagger.feature2input

    if args.fp16:
        supertagger.half()
    supertagger.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        supertagger = DDP(supertagger)
    elif n_gpu > 1:
        supertagger = torch.nn.DataParallel(supertagger)

    supertagger.to(device)

    supertagger.eval()

    all_y_true = []
    all_y_pred = []
    output_suppertag_list = []

    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]
        eval_features = convert_examples_to_features(eval_batch_examples)

        input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, \
        dep_adjacency_matrix = feature2input(device, eval_features)

        with torch.no_grad():
            _, logits = supertagger(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask,
                                    adjacency_matrix=dep_adjacency_matrix
                                    )

        logits = F.softmax(logits, dim=2)
        argmax_logits = torch.argmax(logits, dim=2)
        argsort_loagits = torch.argsort(logits, dim=2, descending=True)
        argmax_logits = argmax_logits.detach().cpu().numpy()
        argsort_loagits = argsort_loagits.detach().cpu().numpy()[:, :, : clipping_top_n]
        logits = logits.to('cpu').numpy()
        label_ids = label_ids.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    all_y_true.append(temp_1)
                    all_y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(id2label[label_ids[i][j]])
                    temp_2.append(id2label[argmax_logits[i][j]])

        for i in range(len(label_ids)):
            ex = eval_batch_examples[i]
            label = label_ids[i]
            text = ex.text_a.split(' ')
            output_line = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    assert len(text) == j - 1
                    output_suppertag_list.append('#word#'.join(output_line))
                    break
                else:
                    super_tag_str_list = []
                    prob_str_list = []
                    for tag_id in argsort_loagits[i][j]:
                        if tag_id == 0:
                            continue
                        label = id2label[tag_id]
                        prob = logits[i][j][tag_id]
                        if len(super_tag_str_list) > 0 and prob < clipping_threshold:
                            break
                        else:
                            super_tag_str_list.append(label)
                            prob_str_list.append(str(prob))
                    word_str = text[j - 1] + '\t' + '#'.join(
                        super_tag_str_list) + '\t' + '#'.join(prob_str_list)
                    output_line.append(word_str)

    y_true_all = []
    y_pred_all = []
    eval_sentence_all = []
    for y_true_item in all_y_true:
        y_true_all += y_true_item
    for y_pred_item in all_y_pred:
        y_pred_all += y_pred_item

    acc = evaluator.supertag_acc(y_pred_all, y_true_all)

    for example, y_true_item in zip(eval_examples, all_y_true):
        sen = example.text_a
        sen = sen.strip()
        sen = sen.split(' ')
        if len(y_true_item) != len(sen):
            # print(len(sen))
            sen = sen[:len(y_true_item)]
        eval_sentence_all.append(sen)

    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')

    correct_results_file = os.path.join('./tmp', 'test.correct.result.txt')

    with open(correct_results_file, 'w', encoding='utf8') as f:
        for index, (sen, y_true, y_pred) in enumerate(zip(eval_sentence_all, all_y_true, all_y_pred)):
            correct = True
            for y_t, y_p in zip(y_true, y_pred):
                if not y_t == y_p:
                    correct = False
                    break
            if correct and len(sen) < 20:
                f.write('ID=%d\n' % (index + 1))
                f.write(' '.join(sen) + '\n')
                for w, y_t in zip(sen, y_true):
                    f.write('%s\t%s\n' % (w, y_t))
                f.write('\n')

    auto_output_file = os.path.join('./tmp', 'test.auto')

    supertag_output_file = os.path.join('./tmp', 'test.supertag.txt')

    with open(supertag_output_file, 'w', encoding='utf8') as f:
        for line in output_suppertag_list:
            f.write(line + '\n')

    command = 'java -jar ' + ccgparse + ' -f ' + supertag_output_file + ' -o ' + auto_output_file + ' >' + auto_output_file
    print(command)
    subprocess.run(command, shell=True)

    dep_output_file = os.path.join('./tmp', 'test.dep')

    command = './auto2dep.sh ' + candc_path + ' ' + auto_output_file + ' ' + dep_output_file
    print(command)
    subprocess.run(command, shell=True)

    eval_output_file = os.path.join('./tmp', 'test.eval')
    if args.eval_data_path.find('dev') > -1:
        tag_gold = os.path.join(args.eval_data_dir, 'gold_files', 'dev.stagged')
        dep_gold = os.path.join(args.eval_data_dir, 'gold_files', 'dev.dep.gold')
    else:
        tag_gold = os.path.join(args.eval_data_dir, 'gold_files', 'test.stagged')
        dep_gold = os.path.join(args.eval_data_dir, 'gold_files', 'test.dep.gold')
    command = 'python ccg_eval.py -r ' + tag_gold + ' ' + dep_gold + ' ' \
              + dep_output_file + ' ' + auto_output_file + ' >' + eval_output_file
    print(command)
    subprocess.run(command, shell=True)

    results = evaluator.eval_file_reader(eval_output_file)

    for key, value in results.items():
        h_key = 'test_' + key
        if h_key in results:
            results[h_key] = value

    results['acc'] = acc

    log_info = []
    for key, value in results.items():
        log_info.append(key)
        log_info.append(str(value))
    info_str = ' '.join(log_info)
    print(info_str)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--dev_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--test_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--eval_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--eval_data_dir",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output path of segmented file")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_xlnet",
                        action='store_true',
                        help="Whether to use XLNet.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")
    parser.add_argument('--ngram_freq_threshold', type=int, default=0, help="The threshold of n-gram frequency")
    parser.add_argument('--max_ngram_length', type=int, default=5,
                        help="The maximum length of n-grams to be considered.")
    parser.add_argument('--model_name', type=str, default=None, help="")
    parser.add_argument("--use_weight", action='store_true', help="")
    parser.add_argument("--use_gcn", action='store_true', help="")
    parser.add_argument("--use_in_chunk", action='store_true', help="")
    parser.add_argument("--use_cross_chunk", action='store_true', help="")
    parser.add_argument('--gcn_layer_number', type=int, default=2,
                        help="The maximum length of n-grams to be considered.")
    parser.add_argument('--clipping_top_n', type=int, default=5, help="Can be used for distant debugging.")
    parser.add_argument('--clipping_threshold', type=float, default=0.0005, help="Can be used for distant debugging.")

    args = parser.parse_args()

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    elif args.do_predict:
        raise ValueError()
        # predict(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    main()
