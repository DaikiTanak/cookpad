
from preprocess import title_ing
import collections
import numpy as np
import chainer
from chainer import Chain, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L


train_title, train_ing, train_step, test_title, test_ing, test_step = title_ing()
print(len(train_title), len(train_ing))

print(train_ing[2])

train_title_list = []
train_ing_list = []
train_step_list = []

title_voc = {}
ing_voc = {}
""" タイトル、材料、レシピを記述したtxtファイルの作成 """
for title, ing, step in zip(train_title, train_ing, train_step):
    t, i = "", ""
    for word in title:
        if(word == "簡単"):
            continue
        if(not word in title_voc.keys()):
            title_voc[word] = 1
        else:
            title_voc[word] += 1
        t += word
        t += " "
    for word in ing:
        if(not word in ing_voc.keys()):
            ing_voc[word] = 1
        else:
            ing_voc[word] += 1
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


#単語vocabraryの作成
title_word_list = []
for title in title_voc.keys():
    title += "\n"
    title_word_list.append(title)
ing_word_list = []
for ing in ing_voc.keys():
    ing += "\n"
    ing_word_list.append(ing)

print("length of vocabrary.")
print(len(title_word_list), len(ing_word_list))
f = open('title_voc.txt', 'w') # 書き込みモードで開く
f.writelines(title_word_list) # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる
f = open('ing_voc.txt', 'w') # 書き込みモードで開く
f.writelines(ing_word_list) # 引数の文字列をファイルに書き込む
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
