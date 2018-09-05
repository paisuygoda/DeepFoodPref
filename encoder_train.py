from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle
import numpy as np
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def get_parser():
    parser = argparse.ArgumentParser(description='hyper params')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('-g', '--gpu', default=0, nargs='+', type=int)
    parser.add_argument('--featDim', default=128, type=int)
    parser.add_argument('--batchSize', default=64, type=int)
    return parser


class FoodSequenceDataset(Dataset):
    def __init__(self, csv_file='data/subdata/user_meals_dataset.p'):
        with open(csv_file, mode='rb') as f:
            self.ds = pickle.load(f)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        user_id = self.ds[idx][0]
        log_len = self.ds[idx][1]
        nutrition_log = self.ds[idx][2]
        return user_id, log_len, nutrition_log


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hidden=False):
        if hidden:
            output, hidden = self.lstm(input, hidden)
        else:
            output, hidden = self.lstm(input)
        return output, hidden


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

def train(original_tensor, tensor_len, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, max_length=61):

    teacher_forcing_ratio = 0.5

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    original_variable = torch.autograd.Variable(original_tensor).cuda()

    loss = 0.0

    encoder_hidden = False
    for i in range(max_length):
        encoder.lstm.flatten_parameters()
        _, encoder_hidden = encoder(torch.autograd.Variable(original_variable.data.narrow(1, i, 1)).cuda(), encoder_hidden)

    decoder_input = torch.autograd.Variable(torch.zeros(batch_size, 1, 32)).cuda()
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for i in range(max_length):
            decoder.lstm.flatten_parameters()
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, original_variable.data.narrow(1, i, 1))
            decoder_input = torch.autograd.Variable(original_variable.data.narrow(1, i, 1)).cuda()  # Teacher forcing

    else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(max_length):
                decoder.lstm.flatten_parameters()
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                decoder_input = decoder_output.detach()  # detach from history as input

                loss += criterion(decoder_output, original_variable.data.narrow(1, i, 1))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / tensor_len


def trainIters(encoder, decoder, dataloader, n_iters, batch_size, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss().cuda()

    for i in range(n_iters):
        # 後でbatchにする・全部回る
        for iter, (user_id, tensor_len, original_tensor) in enumerate(dataloader):

            # tensor_lenの61はデータセット内の最大食事数
            loss = train(original_tensor, 61, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / len(dataloader)),
                                             iter, iter / len(dataloader) * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)

if __name__ == '__main__':
    parser = get_parser()
    param = parser.parse_args()

    encoder_lstm = EncoderLSTM(32, param.featDim).cuda()
    decoder_lstm = DecoderLSTM(param.featDim, 32).cuda()

    dataset = FoodSequenceDataset()
    dataloader = DataLoader(dataset, batch_size=param.batchSize, shuffle=True, num_workers=4)

    trainIters(encoder_lstm, decoder_lstm, dataloader, 10, param.batchSize, print_every=100)

