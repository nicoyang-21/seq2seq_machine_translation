"""
初始数据处理模块，将数据转换为{id:词}
"""
import re

import unicodedata


sos_token = 0  # “Start of sentence”
eos_token = 1  # “End of sentence”   加入文本两端，控制解码过程


class Lang:
    """
    这个类的目的是获取{word: index}, {word: count}, {index, word},
    便于进行后续操作
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 2  # {sos_token, eos_token}

    def index_words(self, sentence):
        if self.name == 'cn':
            for word in sentence:
                self.index_word(word)
        else:
            for word in sentence.split(' '):
                self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):     # 字符标准化
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z\u4e00-\u9fa5.!?，。？]+", r" ", s)
    return s


