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
    return '%s (%s ETA)' % (asMinutes(s), asMinutes(rs))


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
    parser.add_argument('--batchSize', default=128, type=int)
    parser.add_argument('--maxLength', default=61, type=int)
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
        firstday = self.ds[idx][2]
        nutrition_log = self.ds[idx][3]
        return user_id, log_len, firstday, nutrition_log


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

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        pred = self.out(output)
        return output, hidden, pred


def train(original_tensor, tensor_len, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, max_length=61):

    teacher_forcing_ratio = 0.5
    batch_size = original_tensor.size()[0]
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    original_variable = torch.autograd.Variable(original_tensor.float()).cuda()

    loss = 0.0

    encoder_hidden = False
    for i in range(max_length):
        encoder.lstm.flatten_parameters()
        encoder_output, encoder_hidden = encoder(torch.autograd.Variable(original_variable.data.narrow(1, i, 1)).cuda(), encoder_hidden)

    decoder_input = encoder_output
    decoder_hidden = encoder_hidden

    for di in range(max_length):
        decoder.lstm.flatten_parameters()
        decoder_output, decoder_hidden, pred = decoder(decoder_input, decoder_hidden)
        decoder_input = decoder_output.detach()  # detach from history as input

        loss += criterion(pred.view(batch_size, 32),
                          torch.autograd.Variable(original_variable.data.narrow(1, di, 1).contiguous().view(batch_size, 32).float()))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data.cpu().numpy()[0] / tensor_len


def trainEpochs(encoder, decoder, dataloader, n_epoch, batch_size, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().cuda()

    for i in range(1, n_epoch+1):
        epochstart = time.time()
        epoch = 1
        for j, (user_id, tensor_len, firstday, original_tensor) in enumerate(dataloader):
            epoch = j+1
            # tensor_lenの61はデータセット内の最大食事数
            loss = train(original_tensor, 61, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size)
            print_loss_total += loss
            plot_loss_total += loss
            loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d%%) %.4f' % (timeSince(epochstart, epoch / (len(dataloader) + 1)), epoch / (len(dataloader) + 1) * 100, print_loss_avg))

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        print('End of epoch... %s (%d%%) \nLoss: %.4f' % (timeSince(start, i / n_epoch), i / n_epoch * 100, loss_total/epoch))

    showPlot(plot_losses)


def extract_feature(encoder, dataloader, max_length, feat_dim):

    feature_dict = {}
    for j, (user_id, tensor_len, firstday, original_tensor) in enumerate(dataloader):
        batch_size = original_tensor.size()[0]
        original_variable = torch.autograd.Variable(original_tensor.float()).cuda()
        encoder_hidden = False
        for i in range(max_length):
            encoder.lstm.flatten_parameters()
            encoder_output, encoder_hidden = encoder(torch.autograd.Variable(original_variable.data.narrow(1, i, 1)).cuda(),
                                                     encoder_hidden)
        features = encoder_hidden[0].data.cpu().view(batch_size, feat_dim).numpy()
        user_id = user_id.data.cpu().view(batch_size).numpy()
        firstday = firstday.data.cpu().view(batch_size.numpy())
        for user, day, feature in zip(user_id, firstday, features):
            if user in feature_dict:
                feature_dict[user].append((day, feature))
            else:
                feature_dict[user] = [(day, feature)]

    with open("data/subdata/food_pref.p", mode='wb') as f:
        pickle.dump(feature_dict, f)

if __name__ == '__main__':
    parser = get_parser()
    param = parser.parse_args()

    encoder_lstm = EncoderLSTM(32, param.featDim).cuda()
    decoder_lstm = DecoderLSTM(param.featDim, 32).cuda()

    dataset = FoodSequenceDataset()
    dataloader = DataLoader(dataset, batch_size=param.batchSize, shuffle=True, num_workers=4)

    trainEpochs(encoder_lstm, decoder_lstm, dataloader, 1, param.batchSize, print_every=100)

    extract_feature(encoder_lstm, dataloader, param.maxLength, param.featDim)

