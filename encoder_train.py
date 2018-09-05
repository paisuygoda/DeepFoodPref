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
    parser.add_argument('--maxLength', default=9, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    return parser


class FoodSequenceDataset(Dataset):
    def __init__(self, csv_file='data/subdata/user_meals_dataset_FM.p'):
        with open(csv_file, mode='rb') as f:
            self.ds = pickle.load(f)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        user_id = self.ds[idx][0]
        firstday = self.ds[idx][1]
        nutrition_log = self.ds[idx][2]
        return user_id, firstday, nutrition_log


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


def train(original_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=9):

    batch_size = original_tensor.size()[0]
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    original_variable = torch.autograd.Variable(original_tensor.float(), requires_grad=False).cuda()

    loss = 0.0

    encoder_hidden = False
    for i in range(max_length):
        encoder.lstm.flatten_parameters()
        encoder_output, encoder_hidden = encoder(torch.autograd.Variable(original_variable.data.narrow(1, i, 1), requires_grad=False).cuda(), encoder_hidden)

    decoder_input = torch.autograd.Variable(torch.zeros((batch_size, 1, 128)), requires_grad=False).cuda()
    decoder_hidden = encoder_hidden

    for di in range(max_length):
        decoder.lstm.flatten_parameters()
        decoder_output, decoder_hidden, pred = decoder(decoder_input, decoder_hidden)
        decoder_input = decoder_output.detach()  # detach from history as input

        loss += criterion(pred.view(batch_size, 31),
                          torch.autograd.Variable(original_variable.data.narrow(1, di, 1).contiguous().view(batch_size, 31).float(), requires_grad=False))

    if loss.data.cpu().numpy()[0] < 10000.0:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.data.cpu().numpy()[0] / max_length


def trainEpochs(encoder, decoder, dataloader, n_epoch, max_length, print_every=1000, plot_every=100, learning_rate=0.1):
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
        valid_epoch = 0
        for j, (user_id, firstday, original_tensor) in enumerate(dataloader):
            epoch = j+1
            loss = train(original_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
            print_loss_total += loss
            plot_loss_total += loss
            if loss < 10000.0:
                loss_total += loss
                valid_epoch += 1

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d%%) %.4f' % (timeSince(epochstart, epoch / (len(dataloader) + 1)), epoch / (len(dataloader) + 1) * 100, print_loss_avg))

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        if valid_epoch > 0:
            print('End of epoch... %s (%d%%) \nLoss: %.4f' % (timeSince(start, i / n_epoch), i / n_epoch * 100, loss_total/epoch))
            loss_total = 0
    # showPlot(plot_losses)


def extract_feature(encoder, dataloader, max_length, feat_dim):

    feature_dict = {}
    for j, (user_id, tensor_len, firstday, original_tensor) in enumerate(dataloader):
        batch_size = original_tensor.size()[0]
        original_variable = torch.autograd.Variable(original_tensor.float(), requires_grad=False).cuda()
        encoder_hidden = False
        for i in range(max_length):
            encoder.lstm.flatten_parameters()
            encoder_output, encoder_hidden = encoder(torch.autograd.Variable(original_variable.data.narrow(1, i, 1), requires_grad=False).cuda(),
                                                     encoder_hidden)
        features = encoder_output.data.cpu().view(batch_size, feat_dim).numpy()
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

    encoder_lstm = EncoderLSTM(31, param.featDim).cuda()
    decoder_lstm = DecoderLSTM(param.featDim, 31).cuda()

    dataset = FoodSequenceDataset()
    dataloader = DataLoader(dataset, batch_size=param.batchSize, shuffle=True, num_workers=4)

    trainEpochs(encoder_lstm, decoder_lstm, dataloader, param.epoch, param.maxLength, print_every=100)

    extract_feature(encoder_lstm, dataloader, param.maxLength, param.featDim)

