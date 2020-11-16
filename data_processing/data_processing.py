import os
from os import path
import re
import argparse
from collections import defaultdict

OUTPUT_HOME = '../data'

if not os.path.exists(OUTPUT_HOME):
    os.mkdir(OUTPUT_HOME)

# we ignore these sentences because they will cause errors when running evaluation code from C&C parser.
auto_ignore_id = {'wsj_0041.26', 'wsj_0041.37', 'wsj_0052.1', 'wsj_0052.5', 'wsj_0056.1', 'wsj_0056.5',
                  'wsj_0062.50', 'wsj_0085.54', 'wsj_0089.3', 'wsj_0090.24', 'wsj_2330.38', 'wsj_2366.47',
                  'wsj_2373.11', 'wsj_2377.42', 'wsj_2385.27', 'wsj_2385.29', 'wsj_2385.33', 'wsj_2385.37',
                  'wsj_2391.1'}


class CCGSupertagging:
    def __init__(self, ccgbank_home):
        self.auto_dir = path.join(ccgbank_home, 'RAW', 'data', 'AUTO')
        self.parg_dir = path.join(ccgbank_home, 'RAW', 'data', 'PARG')
        self.processed_dir = OUTPUT_HOME
        self.splits = {'train': ['%02d' % index for index in range(2, 22)],
                       'dev': ['00'],
                       'test': ['23']}

    def process_supertag(self):
        for flag, section_list in self.splits.items():
            output_file = path.join(self.processed_dir, flag + '.tsv')
            with open(output_file, 'w', encoding='utf8') as fw:
                for input_section in section_list:
                    input_section_path = path.join(self.auto_dir, input_section)
                    input_file_list = [file_name for file_name in os.listdir(input_section_path)
                                       if file_name.endswith('.auto')]
                    input_file_list.sort()
                    for input_file_name in input_file_list:
                        input_file = path.join(input_section_path, input_file_name)
                        sentence_id = ''
                        with open(input_file, 'r', encoding='utf8') as fr:
                            lines = fr.readlines()
                            for line in lines:
                                line = line.strip()
                                if line.startswith('ID='):
                                    sentence_id = line.split()[0][3:]
                                if line.startswith('(<') and sentence_id not in auto_ignore_id:
                                    leaves = re.findall("<L (.*?)>", line)
                                    for leave in leaves:
                                        items = leave.split()
                                        assert len(items) == 5
                                        ccg_categories = items[0]
                                        modified_pos_tags = items[1]
                                        original_pos_tags = items[2]
                                        tokens = items[3]
                                        predicate_arg_categories = items[4]
                                        output_str = tokens + '\t' + \
                                                     modified_pos_tags + '\t' + original_pos_tags + '\t' + \
                                                     ccg_categories + '\t' + predicate_arg_categories + '\n'
                                        fw.write(output_str)
                                    fw.write('\n')

    def prepare_auto_gold(self):
        output_dir = path.join(self.processed_dir, 'tmp')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for flag, section_list in self.splits.items():
            if flag == 'train':
                continue
            output_file = path.join(output_dir, flag + '.auto')
            with open(output_file, 'w', encoding='utf8') as fw:
                for input_section in section_list:
                    input_section_path = path.join(self.auto_dir, input_section)
                    input_file_list = [file_name for file_name in os.listdir(input_section_path)
                                       if file_name.endswith('.auto')]
                    input_file_list.sort()
                    for input_file_name in input_file_list:
                        input_file = path.join(input_section_path, input_file_name)
                        sentence_id = ''
                        with open(input_file, 'r', encoding='utf8') as fr:
                            lines = fr.readlines()
                            for line in lines:
                                line = line.strip()
                                if line.startswith('ID='):
                                    sentence_id = line.split()[0][3:]
                                if sentence_id in auto_ignore_id:
                                    continue
                                else:
                                    fw.write(line)
                                    fw.write('\n')

    def process_dep(self):
        output_dir = path.join(self.processed_dir, 'gold_files')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for flag, section_list in self.splits.items():
            if flag == 'train':
                continue
            output_file = path.join(output_dir, flag + '.dep.gold')
            sent_num = 0
            with open(output_file, 'w', encoding='utf8') as fw:
                for input_section in section_list:
                    input_section_path = path.join(self.parg_dir, input_section)
                    input_file_list = [file_name for file_name in os.listdir(input_section_path)
                                       if file_name.endswith('.parg')]
                    input_file_list.sort()
                    for input_file_name in input_file_list:
                        input_file = path.join(input_section_path, input_file_name)
                        with open(input_file, 'r', encoding='utf8') as fr:
                            lines = fr.readlines()
                            dep_num = 0
                            for line in lines:
                                line = line.strip()
                                if line == '':
                                    continue
                                if line.startswith('<s'):
                                    continue
                                elif line.startswith('<\\s'):
                                    if dep_num > 0:
                                        fw.write('\n')
                                        sent_num += 1
                                        dep_num = 0
                                else:
                                    fields = re.split('\\s+', line)
                                    arg_index, pred_index, cat, slot, arg, pred = fields[:6]
                                    arg_index = int(arg_index) + 1
                                    pred_index = int(pred_index) + 1
                                    dep_num += 1
                                    out_line = "%s_%d %s %s %s_%d" % (pred, pred_index, cat, slot, arg, arg_index)
                                    fw.write(out_line)
                                    fw.write('\n')

    def get_tagset(self):
        ccg2count = defaultdict(int)
        flag = 'train'
        sent_num = 0
        max_sent_len = -1
        token2count = defaultdict(int)
        input_file = path.join(self.processed_dir, flag + '.tsv')
        with open(input_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            sent_len = 0
            for line in lines:
                line = line.strip()
                if line == '':
                    sent_num += 1
                    if sent_len > max_sent_len:
                        max_sent_len = sent_len
                    sent_len = 0
                    continue
                splits = line.split()
                token = splits[0]
                ccg = splits[3]
                token2count[token] += 1
                ccg2count[ccg] += 1
                sent_len += 1
        # print('sentences in %s: %d' % (flag, sent_num))
        # print('word types in %s: %d' % (flag, len(token2count)))
        # print('words in %s: %d' % (flag, sum(token2count.values())))
        # print('max sentence length: %d' % max_sent_len)
        sorted_ccg2count = sorted(ccg2count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        eval_ccg = [ccg2count[0] for ccg2count in sorted_ccg2count if ccg2count[1] >= 10]
        print(len(eval_ccg))
        assert len(eval_ccg) == 425
        eval_label2id_file = path.join(self.processed_dir, '425tags')
        with open(eval_label2id_file, 'w', encoding='utf8') as f:
            for ccg in eval_ccg:
                f.write(ccg + '\n')


def main(args):
    ccg_supertagger = CCGSupertagging(args.ccgbank_home)
    if args.supertag:
        print('Generating files for CCG supertagging...')
        ccg_supertagger.process_supertag()
        ccg_supertagger.get_tagset()
    if args.ccg_parsing:
        print('Generating files for CCG parsing...')
        ccg_supertagger.prepare_auto_gold()
        ccg_supertagger.process_dep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ccgbank_home",
                        required=True,
                        type=str,
                        help="The home directory of CCGbank.")
    parser.add_argument("--supertag",
                        action='store_true',
                        help="Generate files for CCG supertagging.")
    parser.add_argument("--ccg_parsing",
                        action='store_true',
                        help="Generate files for CCG parsing.")

    args = parser.parse_args()

    main(args)
