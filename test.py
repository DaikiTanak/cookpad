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

model = Seq2seq(3, 7334, 6829, 1024)
serializers.load_npz("LSTM.model", model)
print("model loaded.")


f = open('test_title.txt', 'r') # 書き込みモードで開く
target = f.readlines() # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる
f = open('test_ing.txt', 'r') # 書き込みモードで開く
source = f.readlines() # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる

target =  list(map(lambda x: x[:-2], target))
source =  list(map(lambda x: x[:-2], source))

for source_sentence in source:
    result = model.translate([model.xp.array(source_sentence)])[0]

    result_sentence = ' '.join([target_words[y] for y in result])
    print(result_sentence)

print(target[:3], source[:3])
