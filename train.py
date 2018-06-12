from model import LSTM
from preprocess import title_ing
import collections
import numpy as np
import chainer
from chainer import Chain, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L


train_title, train_ing, test_title, test_ing = title_ing()
print(len(train_title), len(test_ing))

train_title_list = []
train_ing_list = []

title_voc = []
ing_voc = []
for title, ing in zip(train_title, train_ing):
    t, i = "", ""
    for word in title:
        t += word
        t += " "
    for word in ing:
        i += word
        i += " "
    t += "\n"
    i += "\n"

    train_title_list.append(t)
    train_ing_list.append(i)

f = open('train_title.txt', 'w') # 書き込みモードで開く
f.writelines(train_title_list) # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる
f = open('train_ing.txt', 'w') # 書き込みモードで開く
f.writelines(train_ing_list) # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる







test_title_list = []
test_ing_list = []
for title, ing in zip(test_title, test_ing):
    t, i = "", ""
    for word in title:
        t += word
        t += " "
    for word in ing:
        i += word
        i += " "
    t += "\n"
    i += "\n"

    test_title_list.append(t)
    test_ing_list.append(i)

f = open('test_title.txt', 'w') # 書き込みモードで開く
f.writelines(test_title_list) # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる
f = open('test_ing.txt', 'w') # 書き込みモードで開く
f.writelines(test_ing_list) # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる




#[材料、レシピ名]のリスト
train, test = [], []
for title, ing in zip(train_title, train_ing):
    train.append([ing, title])
for title, ing in zip(test_title, test_ing):
    test.append([ing, title])



# 単語辞書
word2id = collections.defaultdict(lambda: len(word2id))
for t in train_title:
    for food in t:
        word2id[food]
for i in train_ing:
    for food in i:
        word2id[food]
for t in test_title:
    for food in t:
        word2id[food]
for i in test_ing:
    for food in i:
        word2id[food]
print(len(word2id))
vocab_size = len(word2id)



model = LSTM(vocab_size, 100, 100, vocab_size)
# GPU対応
#chainer.cuda.get_device(0).use()
#model.to_gpu()
