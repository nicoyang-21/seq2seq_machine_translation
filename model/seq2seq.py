"""
组件模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_preprocess.data_process import USE_CUDA


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        if USE_CUDA:
            hidden = hidden.cuda()
        return hidden


class AttentionTDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layer=1, dropout_p=0.1):
        super(AttentionTDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layer
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layer, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, output_en):
        word_embedded = self.embedding(word_input).view(1, 1, -1)
        output_en = output_en.view(1, self.hidden_size, -1)
        att_weight = F.softmax(torch.bmm(word_embedded, output_en), 1).view(1, -1, 1)
        att_tensor = torch.bmm(output_en, att_weight).view(1, 1, -1)
        word_embedded = torch.add(word_embedded, att_tensor)
        rnn_output, hidden = self.rnn(word_embedded, last_hidden)

        rnn_output = rnn_output.squeeze(0)
        output = F.log_softmax(self.out(rnn_output))

        return output, hidden

