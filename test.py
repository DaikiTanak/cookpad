import argparse
import datetime

from nltk.translate import bleu_score
import numpy
import progressbar
import six

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import MinValueTrigger

from chainer import serializers
from seq2seq import Seq2seq

UNK = 0
EOS = 1

#model = Seq2seq(3, 7334, 6829, 1024)
#model = Seq2seq(3, 5838, 6829, 1024)
model = Seq2seq(3, 21148, 6829, 1024)
serializers.load_npz("LSTM_step2title.model", model)
#serializers.load_npz("LSTM_ing2title2.model", model)

print("model LSTM_step2title loaded.")


gpu = 3
if gpu >= 0:
    chainer.backends.cuda.get_device(gpu).use()
    model.to_gpu(gpu)

f = open('test_title.txt', 'r') # 書き込みモードで開く
target = f.readlines() # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる
f = open('test_ing.txt', 'r') # 書き込みモードで開く
source = f.readlines() # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる

target =  list(map(lambda x: x[:-2], target))
source =  list(map(lambda x: x[:-2], source))


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids

# データをロードする
def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK)
                                 for w in words], numpy.int32)
            data.append(array)
    return data

def main(model):
    #source_ids = load_vocabulary("ing_voc.txt")
    source_ids = load_vocabulary("step_voc.txt")
    target_ids = load_vocabulary("title_voc.txt")

    #test_source = load_data(source_ids, "test_ing.txt")
    test_source = load_data(source_ids, "test_step.txt")
    test_target = load_data(target_ids, "test_title.txt")
    test_data = list(six.moves.zip(test_source, test_target))
    test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}
    res = []
    bar = progressbar.ProgressBar()

    copus_score = 0
    for data in bar(test_data, max_value=len(test_data)):
        ing, title = data

        result = model.translate([model.xp.array(ing)])[0]
        source_sentence = ' '.join([source_words[x] for x in ing])
        target_sentence = ' '.join([target_words[y] for y in title])
        result_sentence = ' '.join([target_words[y] for y in result])
        s = '# source : ' + source_sentence + "\n"
        r = '# result : ' + result_sentence + "\n"
        e = '# expect : ' + target_sentence + "\n"
        #print('# result : ' + result_sentence)
        #print('# expect : ' + target_sentence)
        res.append(s)
        res.append(r)
        res.append(e)

        """ Evaluation """

        score = 0
        for true_id in title:
            if true_id in result:
                score += 1
        #再現できたものの割合
        final_score = score/len(title)
        copus_score += final_score

    f = open('pred_result_step2title.txt', 'w')
    #f = open('pred_true2.txt', 'w')
    f.writelines(res)
    f.close()
    print(copus_score)






if __name__ == "__main__":
    main(model)
