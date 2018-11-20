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


class FoodPrefDataset(Dataset):
    def __init__(self, csv_file='results/preffeat_LSTM_FM_3_days_3_parts_all_nut.p'):
        with open(csv_file, mode='rb') as f:
            self.base = pickle.load(f)
        with open('data/subdata/user_attribute.p', mode='rb') as f:
            self.att = pickle.load(f)
        self.ds = []
        for key, d_list in self.base.items():
            (gender, age, birthday) = self.att[key]
            for (day, feat) in d_list:
                self.ds.append((key, day, feat, gender, age))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        user_id = self.ds[idx][0]
        firstday = self.ds[idx][1]
        feat = self.ds[idx][2]
        gender = self.ds[idx][3]
        age = self.ds[idx][4]
        return feat, user_id, gender, age, firstday


class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.gender = nn.Linear(input_size, 2)
        self.age = nn.Linear(input_size, 7)
        self.user = nn.Linear(input_size, 100)

    def forward(self, input):
        gender_guess = self.gender(input)
        age_guess = self.age(input)
        return gender_guess, age_guess


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

        addloss = criterion(pred.view(batch_size, 31),
                          torch.autograd.Variable(original_variable.data.narrow(1, di, 1).contiguous().view(batch_size, 31).float(), requires_grad=False))

        # if addloss > 1.0:
        #     print(addloss)
        loss += addloss

    lossval = loss.data.cpu().numpy()
    if lossval < 10000.0:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return lossval / max_length


def trainEpochs(eval, dataloader, n_epoch, max_length, learning_rate=0.01, rate_decay=0.95):
    start = time.time()
    cur_lr = learning_rate

    optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().cuda()

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

    dataset = FoodPrefDataset('results/preffeat_LSTM_FM_3_days_3_parts_all_nut.p')
    feat_size = dataset[0][0].shape()[1]
    dataloader = DataLoader(dataset, batch_size=param.batchSize, shuffle=True, num_workers=4)
    eval = Classifier(feat_size)

    trainEpochs(eval, dataloader, param.epoch, param.maxLength, learning_rate=param.lr, rate_decay=param.rateDecay)

    extract_feature(encoder_lstm, dataloader, param.maxLength, param.featDim)

