import math
import random
import time

import torch

from data_preprocess.data_process import MAX_LENGTH, sos_token, USE_CUDA, eos_token

teacher_forcing_ratio = 0.5
clip = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    """
    Zero gradients of both optimizers
    :param input_variable: 输入张量
    :param target_variable: 目标词张量
    :param encoder: EnconderRNN
    :param decoder: DecoderRNN
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param criterion:
    :param max_length:
    :return:
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # 获取输入和目标句子长度
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # 通过encoder运行输入
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # 准备输入和输出张量
    decoder_input = torch.LongTensor([[sos_token]])
    decoder_hidden = encoder_hidden  # 使用encoder的最后的hidden做为decoder初始hidden
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # 选择是否使用教师指导
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:  # 使用真实的目标此作为输入
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # 目标的下一个输入词

    else:  # 使用网络的预测输出作为下一时刻的输出
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            topv, topi = decoder_output.data.topk(1)  # 获取输出最可能结果的index
            ni = topi[0][0]

            decoder_input = torch.LongTensor([[ni]])  # 作为下一时刻的输入
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            if ni == eos_token:
                break
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# 辅助函数
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))





