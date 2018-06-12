import re
import numpy as np
import chainer
from chainer import Chain, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

class LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 中間ベクトルのサイズ
        """
        super(LSTM_Decoder, self).__init__(
            # 入力された単語を単語ベクトルに変換する層
            ye = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 単語ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
            eh = L.Linear(embed_size, 4 * hidden_size),
            # 中間ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
            hh = L.Linear(hidden_size, 4 * hidden_size),
            # 出力されたベクトルを単語ベクトルのサイズに変換する層
            he = L.Linear(hidden_size, embed_size),
            # 単語ベクトルを語彙サイズのベクトル（one-hotなベクトル）に変換する層
            ey = L.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        """

        :param y: one-hotなベクトル
        :param c: 内部メモリ
        :param h: 中間ベクトル
        :return: 予測単語、次の内部メモリ、次の中間ベクトル
        """
        # 入力された単語を単語ベクトルに変換し、tanhにかける
        e = F.tanh(self.ye(y))
        # 内部メモリ、単語ベクトルの4倍+中間ベクトルの4倍をLSTMにかける
        c, h = F.lstm(c, self.eh(e) + self.hh(h))
        # 出力された中間ベクトルを単語ベクトルに、単語ベクトルを語彙サイズの出力ベクトルに変換
        t = self.ey(F.tanh(self.he(h)))
        return t, c, h
