from model import LSTM
from preprocess import title_ing
import collections

train_title, train_ing, test_title, test_ing = title_ing()
print(len(train_title), len(test_ing))

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
