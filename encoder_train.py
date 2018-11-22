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

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %2ds' % (m, s)


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
    parser.add_argument('--batchSize', default=512, type=int)
    parser.add_argument('--maxLength', default=9, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--rateDecay', default=0.95, type=float)
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
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input):
        output, hidden = self.lstm(input.transpose_(0, 1))
        return output, hidden


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.fin = nn.ReLU()

    def forward(self, input, hidden):
        output, hidden = self.lstm(input.transpose_(0,1), hidden)
        pred = self.out(output)
        pred = self.fin(pred)
        return output, hidden, pred


def train(original_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=9):

    batch_size = original_tensor.size()[0]
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    original_variable = torch.autograd.Variable(original_tensor.float(), requires_grad=False).cuda()

    loss = 0.0

    encoder_output, encoder_hidden = encoder(original_variable)

    decoder_input = torch.autograd.Variable(torch.zeros((batch_size, 1, 128)), requires_grad=False).cuda()
    decoder_hidden = encoder_hidden

    for di in range(max_length):
        decoder.lstm.flatten_parameters()
        decoder_output, decoder_hidden, pred = decoder(decoder_input, decoder_hidden)
        decoder_input = decoder_output.detach()  # detach from history as input

        addloss = criterion(pred.view(batch_size, 31),
                          torch.autograd.Variable(original_variable.data.narrow(0, di, 1).contiguous().view(batch_size, 31).float(), requires_grad=False))

        # if addloss > 1.0:
        #     print(addloss)
        loss += addloss

    lossval = loss.data.cpu().numpy()
    if lossval < 10000.0:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return lossval / max_length


def trainEpochs(encoder, decoder, dataloader, n_epoch, max_length, learning_rate=0.01, rate_decay=0.95):
    start = time.time()
    cur_lr = learning_rate

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().cuda()

    for i in range(1, n_epoch+1):
        for j, (user_id, firstday, original_tensor) in enumerate(dataloader):
            loss = train(original_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)

        cur_lr = cur_lr * rate_decay
        for param_group in encoder_optimizer.param_groups:
            param_group['lr'] = cur_lr
        for param_group in decoder_optimizer.param_groups:
            param_group['lr'] = cur_lr

        print('Epoch %3d: %s (%d%%) \tLoss: %.4f' % (i, timeSince(start, i / n_epoch), i / n_epoch * 100, loss))


def extract_feature(encoder, datloader, max_length, feat_dim):

    feature_dict = {}
    for j, (user_id, firstday, original_tensor) in enumerate(datloader):
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

    trainEpochs(encoder_lstm, decoder_lstm, dataloader, param.epoch, param.maxLength, learning_rate=param.lr, rate_decay=param.rateDecay)

    extract_feature(encoder_lstm, dataloader, param.maxLength, param.featDim)

