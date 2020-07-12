import sys
import re

VERBOSE = False
RELATIONS = False

arg = 1
while arg < len(sys.argv):
    if sys.argv[arg][0] != '-':
        break
    if sys.argv[arg] == '-v':
        VERBOSE = True
    elif sys.argv[arg] == '-r':
        RELATIONS = True
    arg += 1

(GOLD_LEXCATS, GOLD_DEPS, TEST, AUTO) = sys.argv[arg:]

eval_length = 0

COMMAND_LINE = ' '.join(sys.argv)

IGNORE = set(map(lambda x: tuple(x.split()), filter(lambda x: not x.startswith('#'), r"""
rule_id 7
rule_id 11
rule_id 12
rule_id 14
rule_id 15
rule_id 16
rule_id 17
rule_id 51
rule_id 52
rule_id 56
rule_id 91
rule_id 92
rule_id 95
rule_id 96
rule_id 98
conj 1 0
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 0
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 2
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 3
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 6
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 9
((S[b]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 6
((S[b]{_}\NP{Y}<1>){_}/PP{Z}<2>){_} 1 6
(((S[b]{_}\NP{Y}<1>){_}/PP{Z}<2>){_}/NP{W}<3>){_} 1 6
(S[X]{Y}/S[X]{Y}<1>){_} 1 13
(S[X]{Y}/S[X]{Y}<1>){_} 1 5
(S[X]{Y}/S[X]{Y}<1>){_} 1 55
((S[X]{Y}/S[X]{Y}){Z}\(S[X]{Y}/S[X]{Y}){Z}<1>){_} 2 97
((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_} 2 4
((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_} 2 93
((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_} 2 8
((S[X]{Y}\NP{Z}){Y}/(S[X]{Y}<1>\NP{Z}){Y}){_} 2 94
((S[X]{Y}\NP{Z}){Y}/(S[X]{Y}<1>\NP{Z}){Y}){_} 2 18
been ((S[pt]{_}\NP{Y}<1>){_}/(S[ng]{Z}<2>\NP{Y*}){Z}){_} 1 0
been ((S[pt]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 there 0
been ((S[pt]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 There 0
be ((S[b]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 there 0
be ((S[b]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 There 0
been ((S[pt]{_}\NP{Y}<1>){_}/(S[pss]{Z}<2>\NP{Y*}){Z}){_} 1 0
been ((S[pt]{_}\NP{Y}<1>){_}/(S[adj]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[pss]{Z}<2>\NP{Y*}){Z}){_} 1 0
have ((S[b]{_}\NP{Y}<1>){_}/(S[pt]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[adj]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[ng]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[pss]{Z}<2>\NP{Y*}){Z}){_} 1 0
going ((S[ng]{_}\NP{Y}<1>){_}/(S[to]{Z}<2>\NP{Y*}){Z}){_} 1 0
have ((S[b]{_}\NP{Y}<1>){_}/(S[to]{Z}<2>\NP{Y*}){Z}){_} 1 0
Here (S[adj]{_}\NP{Y}<1>){_} 1 0
# this is a dependency Julia doesn't have but looks okay
from (((NP{Y}\NP{Y}<1>){_}/(NP{Z}\NP{Z}){W}<3>){_}/NP{V}<2>){_} 1 0
""".strip().split('\n'))))

deps_ignored = 0


def ignore(pred, cat, slot, arg, rule_id):
    global deps_ignored
    res = ('rule_id', rule_id) in IGNORE or \
          (cat, slot, rule_id) in IGNORE or \
          (pred, cat, slot, rule_id) in IGNORE or \
          (pred, cat, slot, arg, rule_id) in IGNORE
    deps_ignored += res
    return res


MARKUP = re.compile(r'<[0-9]>|\{[A-Z_]\*?\}|\[X\]')


def strip_markup(cat):
    cat = MARKUP.sub('', cat)
    if cat[0] == '(':
        return cat[1:-1]
    else:
        return cat


def next_gold_lexcats(lines, index):
    lexcats = []

    # line = file.readline()
    # if not line:
    #   die("unexpected end of file reading gold standard lexical categories")

    line = lines[index]

    for (i, token) in enumerate(line.split()):
        (word, pos, cat) = token.split('|')
        lexcats.append((word, cat))

    return lexcats, index + 1


def next_gold_deps(lines, index):
    deps = set()
    udeps = set()

    line = lines[index]
    if not line:
        return (True, deps, udeps)
    while line:
        line = line.strip()
        if not line:
            break
        (pred, cat, slot, arg) = line.split()
        deps.add((pred, cat, slot, arg))
        udeps.add((pred, arg))
        index += 1
        line = lines[index]

    return (False, deps, udeps), index


def next_test(lines, index):
    lexcats = []
    deps = set()
    udeps = set()
    rule_ids = {}

    # line = file.readline()
    line = lines[index]
    # if not line:
    #   die("unexpected end of file reading gold dependencies")
    if line == '\n':
        return (False, lexcats, deps, udeps, rule_ids), index + 1

    while line:
        line = line.strip()
        if not line or line.startswith('<c>'):
            break
        fields = line.split()
        (pred, cat, slot, arg, rule_id) = fields[:5]
        pred_word = pred.rsplit('_')[0]
        arg_word = arg.rsplit('_')[0]
        if not ignore(pred_word, cat, slot, arg_word, rule_id):
            cat = strip_markup(cat)
            deps.add((pred, cat, slot, arg))
            rule_ids[(pred, cat, slot, arg)] = rule_id
            udeps.add((pred, arg))
        index += 1
        line = lines[index]

    # if not line.startswith('<c>'):
    #   die("unexpected end of file reading test lexical categories")

    for (i, token) in enumerate(line.split()[1:]):
        (word, pos, cat) = token.split('|')
        lexcats.append((word, cat))

    # index += 1
    # line = lines[index]
    # if line != '\n':
    #   die("expected a blank line between each sentence")

    if not rule_ids:
        # No dependencies for this sentence - probably a conversion script error.
        return (False, lexcats, deps, udeps, rule_ids), index + 1

    return (True, lexcats, deps, udeps, rule_ids), index + 1


def get_valid_id():
    id_list = []
    with open(AUTO, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('ID='):
                id_list.append(int(line[3:])-1)
    return id_list


def get_auto():
    pred_auto = []
    with open(AUTO, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('ID='):
                continue
            leaves = re.findall("<L (.*?)>", line)
            pairs = []
            for leave in leaves:
                items = leave.split()
                assert len(items) == 5
                ccg_categories = items[0]
                modified_pos_tags = items[1]
                original_pos_tags = items[2]
                tokens = items[3]
                predicate_arg_categories = items[4]
                pairs.append((tokens, ccg_categories))
            pred_auto.append(pairs)
    return pred_auto


def score_deps(gold_deps, test_deps, rule_ids, verbose, relations,
               correct_relations, incorrect_relations, missing_relations):
    correct = gold_deps.intersection(test_deps)
    if verbose:
        for dep in correct:
            print("correct: %s %s %s %s %s" % (dep + (rule_ids[dep],)))
    if relations:
        for dep in correct:
            correct_relations[dep[1:3]] = correct_relations.setdefault(dep[1:3], 0) + 1

    incorrect = test_deps.difference(gold_deps)
    if verbose:
        for dep in incorrect:
            print("incorrect: %s %s %s %s %s" % (dep + (rule_ids[dep],)))
    if relations:
        for dep in incorrect:
            incorrect_relations[dep[1:3]] = incorrect_relations.setdefault(dep[1:3], 0) + 1

    missing = gold_deps.difference(test_deps)
    if verbose:
        for dep in missing:
            print("missing:   %s %s %s %s ?" % dep)
    if relations:
        for dep in missing:
            missing_relations[dep[1:3]] = missing_relations.setdefault(dep[1:3], 0) + 1

    # if verbose:
    #   print

    return (len(correct), len(incorrect), len(missing))


def score_udeps(gold_deps, test_deps):
    correct = gold_deps.intersection(test_deps)
    incorrect = test_deps.difference(gold_deps)
    missing = gold_deps.difference(test_deps)

    return (len(correct), len(incorrect), len(missing))


def score_lexcats(gold_cats, test_cats):
    if len(gold_cats) != len(test_cats):
        return (0, 0)
        # raise ValueError()

    correct = 0
    for (gold_w, gold_cat), (test_w, test_cat) in zip(gold_cats, test_cats):
        if gold_w != test_w and not test_w.startswith('-'):
            continue
            # raise ValueError()
        if gold_cat == test_cat:
            correct += 1
    return (len(gold_cats), correct)


# def read_preface(filename, file):
#   preface = "# %s was generated using the following commands(s):\n" % filename
#   line = file.readline()
#   while line != "\n":
#     if line.startswith("# this file"):
#       line = file.readline()
#       continue
#     preface += line.replace('# ', '#   ')
#     line = file.readline()
#   return preface

def read_preface(all_lines, index):
    # preface = "# %s was generated using the following commands(s):\n" % filename
    new_index = index
    while all_lines[new_index] != "\n":
        if all_lines[new_index].startswith("# this file"):
            new_index += 1
            continue
        new_index += 1
    return new_index + 1


preface = "# this file was generated by the following command(s):\n"
preface += "# %s\n" % COMMAND_LINE

with open(GOLD_LEXCATS, 'r', encoding='utf8') as f:
    all_gold_lexcats_lines = f.readlines()
    # gold_lexcats_index = 0
    # gold_lexcats_index = read_preface(all_gold_lexcats_lines, gold_lexcats_index)

with open(GOLD_DEPS, 'r', encoding='utf8') as f:
    all_gold_deps_lines = f.readlines()
    # gold_deps_index = 0
    # gold_deps_index = read_preface(all_gold_deps_lines, gold_deps_index)

with open(TEST, 'r', encoding='utf8') as f:
    all_test_lines = f.readlines()
    # gold_text_index = 0
    # gold_text_index = read_preface(all_test_lines, gold_text_index)

nsentences = 0
parse_failures = 0

deps_sent_correct = 0
deps_correct = 0
deps_incorrect = 0
deps_missing = 0

udeps_sent_correct = 0
udeps_correct = 0
udeps_incorrect = 0
udeps_missing = 0

lexcats_sent_correct = 0
lexcats_total = 0
lexcats_correct = 0

relations_correct = {}
relations_incorrect = {}
relations_missing = {}

all_gold_deps_data = []
all_gold_lexcats_data = []
all_test_data = []

first_line_break = True
deps = set()
udeps = set()
for line in all_gold_deps_lines:
    line = line.strip()
    if line.startswith('#'):
        continue

    # if line == '' and first_line_break:
    #     first_line_break = False
    #     continue

    if line == '':
        if len(deps) > 0:
            all_gold_deps_data.append((deps, udeps))
        deps = set()
        udeps = set()
        continue
    (pred, cat, slot, arg) = line.split()
    deps.add((pred, cat, slot, arg))
    udeps.add((pred, arg))

first_line_break = True
for line in all_gold_lexcats_lines:
    line = line.strip()
    if line == '' or line.startswith('#'):
        continue
    if line == '' and first_line_break:
        first_line_break = False
        continue
    lexcats = []
    for (i, token) in enumerate(line.split()):
        (word, pos, cat) = token.split('|')
        lexcats.append((word, cat))
    all_gold_lexcats_data.append(lexcats)

first_line_break = True
lexcats = []
deps = set()
udeps = set()
rule_ids = {}

for line in all_test_lines:
    line = line.strip()
    if line.startswith('#'):
        continue

    if line.startswith('<c>n') and first_line_break:
        first_line_break = False
        continue

    if line.startswith('<c>n'):
        # for (i, token) in enumerate(line.split()[1:]):
        #     (word, pos, cat) = token.split('|')
        #     lexcats.append((word, cat))
        if not rule_ids:
            all_test_data.append((False, lexcats, deps, udeps, rule_ids))
        else:
            all_test_data.append((True, lexcats, deps, udeps, rule_ids))
        lexcats = []
        deps = set()
        udeps = set()
        rule_ids = {}
        continue
    else:
        fields = line.split()
        (pred, cat, slot, arg, rule_id) = fields[:5]
        pred_word = pred.rsplit('_')[0]
        arg_word = arg.rsplit('_')[0]
        if not ignore(pred_word, cat, slot, arg_word, rule_id):
            cat = strip_markup(cat)
            deps.add((pred, cat, slot, arg))
            rule_ids[(pred, cat, slot, arg)] = rule_id
            udeps.add((pred, arg))

    # if not line.startswith('<c>'):
    #   die("unexpected end of file reading test lexical categories")

pred_tags = get_auto()

valid_ids = get_valid_id()
assert len(valid_ids) == len(all_test_data)

assert len(all_gold_lexcats_data) == len(all_gold_deps_data)

all_gold_lexcats_data = [all_gold_lexcats_data[i] for i in valid_ids]
all_gold_deps_data = [all_gold_deps_data[i] for i in valid_ids]

assert len(all_test_data) == len(all_gold_lexcats_data)

for gold_lex, gold_dep, pred_dep, pred_tag in zip(all_gold_lexcats_data, all_gold_deps_data, all_test_data, pred_tags):
    (gold_deps, gold_udeps) = gold_dep

    (parsed, test_lexcats, test_deps, test_udeps, test_rule_ids) = pred_dep

    if len(gold_lex) < eval_length:
        continue

    nsentences += 1
    if not parsed:
        parse_failures += 1
        continue

    (correct, incorrect, missing) = score_deps(gold_deps, test_deps, test_rule_ids, VERBOSE,
                                               RELATIONS, relations_correct, relations_incorrect,
                                               relations_missing)
    deps_correct += correct
    deps_incorrect += incorrect
    deps_missing += missing
    if incorrect == 0 and missing == 0:
        deps_sent_correct += 1

    (correct, incorrect, missing) = score_udeps(gold_udeps, test_udeps)
    udeps_correct += correct
    udeps_incorrect += incorrect
    udeps_missing += missing
    if incorrect == 0 and missing == 0:
        udeps_sent_correct += 1

    (total, correct) = score_lexcats(gold_lex, pred_tag)
    lexcats_total += total
    lexcats_correct += correct
    if total == correct:
        lexcats_sent_correct += 1

# print preface

print("note: all these statistics are over just those sentences")
print("      for which the parser returned an analysis, and ")
print("      dependency extraction script is successful \n")


def pct(val, total):
    if val:
        return 100.0 * val / total
    else:
        return 0.0


def print_acc(name, desc, correct, total):
    acc = pct(correct, total)
    print("%-6s %5.2f%% (%d of %d %s)" % (name + ':', acc, correct, total, desc))


def print_stats(name, correct, incorrect, missing):
    test = correct + incorrect
    prec = pct(correct, test)
    print("%sp:    %5.2f%% (%d of %d %s deps precision)" % (name[0], prec, correct, test, name))

    gold = correct + missing
    recall = pct(correct, gold)
    print("%sr:    %5.2f%% (%d of %d %s deps recall)" % (name[0], recall, correct, gold, name))

    if prec and recall:
        fscore = 2 * prec * recall / (prec + recall)
    else:
        fscore = 0.0
    print("%sf:    %5.2f%% (%s deps f-score)" % (name[0], fscore, name))


def print_rel_stats(relation, correct, incorrect, missing):
    relation = "%s %s" % relation
    test = correct + incorrect
    prec = pct(correct, test)
    gold = correct + missing
    recall = pct(correct, gold)
    if prec and recall:
        fscore = 2 * prec * recall / (prec + recall)
    else:
        fscore = 0.0
    print("%-50s: %6.2f%% %6.2f%% %6.2f%% %6d %6d" % (relation, prec, recall, fscore, test, gold))


# if RELATIONS:
#     relations = relations_correct.copy()
#     for (r, freq) in relations_missing:
#         relations[r] = relations.get(r, 0) + int(freq)
#
#     relations = map(lambda x: (x[1], x[0]), relations.items())
#     relations = sorted(list(relations), key=lambda kv: kv[0])
#
#     for (freq, r) in relations:
#         print_rel_stats(r, relations_correct.get(r, 0), relations_incorrect.get(r, 0),
#                         relations_missing.get(r, 0))
#     print('')

nparsed = nsentences - parse_failures

print_acc("cover", "sentences evaluated  - this includes dependency extraction script errors and parse failures",
          nparsed, nsentences)
print('')
print_acc("cats", "tokens correct", lexcats_correct, lexcats_total)
print_acc("csent", "sentences correct", lexcats_sent_correct, nparsed)
print('')
print_stats("labelled", deps_correct, deps_incorrect, deps_missing)
print_acc("lsent", "labelled deps sentences correct", deps_sent_correct, nparsed)
print('')
print_stats("unlabelled", udeps_correct, udeps_incorrect, udeps_missing)
print_acc("usent", "unlabelled deps sentences correct", udeps_sent_correct, nparsed)
print('')
print_acc("skip", "ignored deps (to ensure compatibility with CCGbank)", deps_ignored,
          deps_correct + deps_incorrect + deps_ignored)
