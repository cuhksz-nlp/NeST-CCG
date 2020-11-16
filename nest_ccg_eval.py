ccgparse = './tag2auto.jar'
candc_path = './candc/'

class Evaluation():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def supertag_acc(self, y_pred, y_true):
        true = 0
        total = 0
        for y_p, y_t in zip(y_pred, y_true):
            if y_t == y_p:
                true += 1
            total += 1
        return 100 * true / total

    def supertag_results(self, y_pred, y, sentence):
        words = sentence.split(' ')
        str_result_list = []

        for word, y_label, y_pred_label in zip(words, y, y_pred):
            str_result_list.append(word + '\t' + y_label + '\t' + y_pred_label + '\n')

        return ''.join(str_result_list)

    def eval_file_reader(self, file_path):
        results = {
            'cover': 0,
            'cats': 0,
            'csent': 0,
            'lp': 0,
            'lr': 0,
            'lf': 0,
            'lsent': 0,
            'up': 0,
            'ur': 0,
            'uf': 0,
            'usent': 0,
            'skip': 0,
        }
        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            splits = line.split()
            key = splits[0][:-1]
            if key in results:
                results[key] = float(splits[1][:-1])
        return results
