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


def train(original_tensor, tensor_len, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=61):

    teacher_forcing_ratio = 0.5

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    original_variable = torch.autograd.Variable(original_tensor).cuda()

    loss = 0.0

    encoder.flatten_parameters()
    encoder_output, encoder_hidden = encoder(original_variable)

    decoder_input = torch.autograd.Variable(torch.zeros(1, 32)).cuda()
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(tensor_len):
            decoder.flatten_parameters()
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, original_variable[di])
            decoder_input = original_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(tensor_len):
            decoder.flatten_parameters()
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.detach()  # detach from history as input

            loss += criterion(decoder_output, original_variable[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / tensor_len


def trainIters(encoder, decoder, dataloader, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
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
            loss = train(original_tensor, 61, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
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

    encoder_lstm = nn.LSTM(32, param.featDim).cuda()
    decoder_lstm = nn.LSTM(param.featDim, 32).cuda()

    dataset = FoodSequenceDataset()
    dataloader = DataLoader(dataset, batch_size=param.batchSize, shuffle=True, num_workers=4)

    trainIters(encoder_lstm, decoder_lstm, dataloader, 10, print_every=100)

