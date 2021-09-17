"""
开始进行训练
"""
import random
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_preprocess.data_process import read_langs, USE_CUDA, variables_from_pair, sos_token, eos_token, prepare_data
from model.seq2seq import EncoderRNN, AttentionTDecoderRNN
from train_function import train, time_since
from data_preprocess.data_process import MAX_LENGTH, variable_from_sentence


def evaluate(sentence, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([[sos_token]])  # SOS
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == eos_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni.item()])

        # Next input is chosen word
        decoder_input = torch.LongTensor([[ni]])
        if USE_CUDA: decoder_input = decoder_input.cuda()

    return decoded_words


def show_plot(points):
    plt.plot([i for i in range(1, 501, 100)], points)
    plt.show()


if __name__ == '__main__':
    hidden_size = 500
    n_layers = 2
    dropout_p = 0.05
    input_lang, output_lang, pairs = prepare_data('cn', 'eng', reverse=False)

    # 实例模型
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
    decoder = AttentionTDecoderRNN(hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

    # 使用GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    # 设置优化函数和损失函数

    learning_rate = 0.0001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # 配置训练参数
    n_epochs = 1
    plot_every = 100
    print_every = 1000

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    # 开始训练
    for epoch in range(1, n_epochs+1):
        print(f'第{epoch}次')
        for pair in tqdm(pairs, desc='Processing'):
            training_pair = variables_from_pair(input_lang, output_lang, pair)  # 从循环中获取训练数据
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
                         criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
                time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    sentence = ['我爱你一万年。', '我給我自己買了書，不是給我妻子。', '你是谁？', '今天出去出大米饭']
    for i in sentence:
        res = evaluate(i, max_length=MAX_LENGTH)
        print(res)
