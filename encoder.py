import re
import numpy as np
import chainer
from chainer import Chain, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 中間層のサイズ
        """
        super(LSTM_Encoder, self).__init__(
            # 単語を単語ベクトルに変換する層
            xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            eh = L.Linear(embed_size, 4 * hidden_size),
            # 出力された中間層を4倍のサイズに変換するための層
            hh = L.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        """
        Encoderの動作
        :param x: one-hotなベクトル
        :param c: 内部メモリ
        :param h: 隠れ層
        :return: 次の内部メモリ、次の隠れ層
        """
        # xeで単語ベクトルに変換して、そのベクトルをtanhにかける
        e = F.tanh(self.xe(x))
        # 前の内部メモリの値と単語ベクトルの4倍サイズ、中間層の4倍サイズを足し合わせて入力
        return F.lstm(c, self.eh(e) + self.hh(h))
