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
    parser.add_argument('--batchSize', default=128, type=int)
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
        self.fin = nn.ReLU()

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        pred = self.out(output)
        pred = self.fin(pred)
        return output, hidden, pred


def train(original_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=9, numnut=31):

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

        addloss = criterion(pred.view(batch_size, numnut),
                          torch.autograd.Variable(original_variable.data.narrow(1, di, 1).contiguous().view(batch_size, numnut).float(), requires_grad=False))

        loss += addloss

    lossval = loss.data.cpu().numpy()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return lossval / max_length


def trainEpochs(encoder, decoder, dataloader, n_epoch, max_length, learning_rate=0.01, rate_decay=0.95, numnut=31):
    cur_lr = learning_rate

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().cuda()

    for i in range(1, n_epoch+1):
        for j, (user_id, firstday, original_tensor) in enumerate(dataloader):
            loss = train(original_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, numnut)
            if loss > 0:
                return 9999.0

        cur_lr = cur_lr * rate_decay
        for param_group in encoder_optimizer.param_groups:
            param_group['lr'] = cur_lr
        for param_group in decoder_optimizer.param_groups:
            param_group['lr'] = cur_lr

    return loss


def extract_feature(encoder, datloader, max_length, feat_dim, outputfile="data/subdata/foodpref.p"):

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
            day_cpu = day.cpu().numpy()
            if user in feature_dict:
                feature_dict[user].append((day_cpu, feature))
            else:
                feature_dict[user] = [(day_cpu, feature)]

    with open(outputfile, mode='wb') as f:
        pickle.dump(feature_dict, f)

if __name__ == '__main__':
    parser = get_parser()
    param = parser.parse_args()

    parts = [1, 3, 6, 8]
    days = [1, 3, 7]
    start = time.time()

    min_loss = 99999.0
    min_i = -1
    for i in range(1000):
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(i)
        filename = "data/subdata/user_meals_dataset_FM_days_3_parts_3_nut_31.p"
        dataset = FoodSequenceDataset(csv_file=filename)
        dataloader = DataLoader(dataset, batch_size=param.batchSize, shuffle=True, num_workers=4)
        encoder_lstm = EncoderLSTM(31, param.featDim).cuda()
        decoder_lstm = DecoderLSTM(param.featDim, 31).cuda()
        finalloss = trainEpochs(encoder_lstm, decoder_lstm, dataloader, 5, 3 * 3,
                                learning_rate=param.lr, rate_decay=param.rateDecay, numnut=31)
        if finalloss < min_loss:
            min_loss = finalloss
            min_i = i

    print(min_i, min_loss)

    for i, part in enumerate(parts):
        for j, day in enumerate(days):

            filename = "data/subdata/user_meals_dataset_FM_days_"+str(day)+"_parts_"+str(part)+"_nut_31.p"
            dataset = FoodSequenceDataset(csv_file=filename)
            dataloader = DataLoader(dataset, batch_size=param.batchSize, shuffle=True, num_workers=4)
            encoder_lstm = EncoderLSTM(31, param.featDim).cuda()
            decoder_lstm = DecoderLSTM(param.featDim, 31).cuda()
            finalloss = trainEpochs(encoder_lstm, decoder_lstm, dataloader, param.epoch, day*part,
                                    learning_rate=param.lr, rate_decay=param.rateDecay, numnut=31)
            print("day: ", day, "\tparts: ", part, "\tall nut\t\tFinal Loss: {0:.4f}\t".format(finalloss),
                  timeSince(start, (((i * 3 + j) * 2) + 1) / 24))
            extract_feature(encoder_lstm, dataloader, day*part, param.featDim,
                            outputfile="results/preffeat_LSTM_FM_"+str(day)+"_days_"+str(part)+"_parts_all_nut.p")

            encoder_lstm = EncoderLSTM(4, param.featDim).cuda()
            decoder_lstm = DecoderLSTM(param.featDim, 4).cuda()
            filename = "data/subdata/user_meals_dataset_FM_days_" + str(day) + "_parts_" + str(part) + "_nut_4.p"
            dataset = FoodSequenceDataset(csv_file=filename)
            dataloader = DataLoader(dataset, batch_size=param.batchSize, shuffle=True, num_workers=4)
            finalloss = trainEpochs(encoder_lstm, decoder_lstm, dataloader, param.epoch, day*part, learning_rate=param.lr,
                        rate_decay=param.rateDecay, numnut=4)
            print("day: ", day, "\tparts: ", part, "\tonly major\tFinal Loss: {0:.4f}\t".format(finalloss),
                  timeSince(start, (((i * 3 + j) * 2) + 2) / 24))
            extract_feature(encoder_lstm, dataloader, day*part, param.featDim,
                            outputfile="results/preffeat_LSTM_FM_" + str(day) + "_days_" + str(
                                part) + "_parts_major_nut.p")

