"""
处理数据的全过程：
读取数据，每一行分别处理，将其转换成句对
对于文本进行处理，过滤无用符号
根据已有文本对于单词进行编号，构建符号到编号的映射
"""
import re

import torch
import unicodedata

sos_token = 0  # “Start of sentence”
eos_token = 1  # “End of sentence”   加入文本两端，控制解码过程
USE_CUDA = True


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
        self.n_words = 2  # 起始{sos_token, eos_token}

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


def unicode_to_ascii(s):  # 字符标准化
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):  # 数据清洗，英文标点.!?前加1个空格，将非字母、汉字和标点替换为空格
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z\u4e00-\u9fa5.!?，。？]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):  # 获取平行预料，并进行清理
    print("Reading lines...")

    # 读取数据并切分成行
    lines = open(f'../data/{lang1}-{lang2}.txt', encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # 反转pairs, 实例化Lang
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    return input_lang, output_lang, pairs


"""过滤句子，控制句子长度"""
MAX_LENGTH = 30


def filter_pair(p):
    return len(p[1].split(' ')) < MAX_LENGTH  # 考虑第二个句子的长度


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse=False)
    print(f"Read {len(pairs)} sentence pairs")

    print('Indexing words..')
    for pair in pairs:
        input_lang.index_word(pair[0])
        output_lang.index_word(pair[1])
    return input_lang, output_lang, pairs


"""
将文本数据转换为张量
为了训练，我们需要将句子变成神经网络可以理解的东西（数字）。每个句子将被分解成单词，然后变成张量，其中每个单词都被索引替换（来自之前的Lang索引）。在创建这些张量时，我们还将附加EOS令牌以表示该句子已结束。
"""


# 返回句子中每个词索引的列表

def indexes_from_sentence(lang, sentence):
    if lang.name == 'cn':
        return [lang.word2index[word] for word in sentence]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(eos_token)
    var = torch.LongTensor(indexes).view(-1, 1)
    if USE_CUDA:
        var = var.cuda()
    return var


def variables_from_pair(input_lang, output_lang, pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)
