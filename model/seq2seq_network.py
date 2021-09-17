import random

import torch
import torch.nn as nn

from data_preprocess.data_process import MAX_LENGTH, sos_token, USE_CUDA, eos_token

clip = 0.5


def seq2seq_network(input_variable, target_variable, encoder, attention_decoder,
                    encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
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
    teacher_forcing_ratio = 0.5
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = attention_decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # 目标的下一个输入词
    else:  # 使用网络的预测输出作为下一时刻的输出
        for di in range(target_length):
            decoder_output, decoder_hidden = attention_decoder(decoder_input, decoder_hidden, encoder_outputs)
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
    torch.nn.utils.clip_grad_norm(attention_decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
