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
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--rateDecay', default=0.95, type=float)
    return parser


def tri_dataloader(file, batch_size):
    train_dataset = FoodSequenceDataset(file + "_train.p")
    val_dataset = FoodSequenceDataset(file + "_val.p")
    test_dataset = FoodSequenceDataset(file + "_test.p")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


class FoodSequenceDataset(Dataset):
    def __init__(self, csv_file='data/subdata/user_meals_dataset_FM.p'):
        with open(csv_file, mode='rb') as f:
            self.ds = pickle.load(f)
        with open('data/subdata/user_attribute.p', mode='rb') as f:
            self.att = pickle.load(f)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        user_id = self.ds[idx][0]
        gender = self.att[user_id][0]
        age = self.att[user_id][1]
        firstday = self.ds[idx][1]
        nutrition_log = self.ds[idx][2]
        return user_id, firstday, nutrition_log, gender, age


class E2E(nn.Module):
    def __init__(self, input_size, hidden_size, isMLP, max_length):
        super(E2E, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size)
        if isMLP:
            self.classifier = MLP(hidden_size)
        else:
            self.classifier = Classifier(hidden_size)
        self.max_length = max_length

    def forward(self, input):
        encoder_hidden = False
        for i in range(self.max_length):
            self.encoder.lstm.flatten_parameters()
            encoder_output, encoder_hidden = self.encoder(torch.autograd.Variable(input.data.narrow(1, i, 1),
                                                                                  requires_grad=False).cuda(), encoder_hidden)
        gender_guess, age_guess = self.classifier(encoder_hidden)
        return gender_guess, age_guess


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


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.linear3 = nn.Linear(input_size, input_size)
        self.gender = nn.Linear(input_size, 2)
        self.age = nn.Linear(input_size, 7)
        self.user = nn.Linear(input_size, 100)

    def forward(self, input):
        mid_feat = self.linear1(input)
        mid_feat = self.linear2(mid_feat)
        mid_feat = self.linear3(mid_feat)
        gender_guess = self.gender(mid_feat)
        age_guess = self.age(mid_feat)
        return gender_guess, age_guess


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hidden=False):
        if hidden:
            output, hidden = self.lstm(input, hidden)
        else:
            output, hidden = self.lstm(input)
        return output, hidden


def train(original_tensor, network, optimizer, gender_criterion, age_criterion):

    optimizer.zero_grad()
    batch_size = original_tensor.size()[0]
    original_variable = torch.autograd.Variable(original_tensor.float(), requires_grad=False).cuda()
    loss = 0.0

    gender_guess, age_guess = network(original_variable)

    gender_loss = gender_criterion(gender_guess, gender)
    age_loss = age_criterion(age_guess, age)

    loss = gender_loss + age_loss

    loss.backward()
    optimizer.step()

    return loss.data.cpu().numpy()


def trainEpochs(network, dataloader, n_epoch, learning_rate=0.01, rate_decay=0.9):
    cur_lr = learning_rate

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    gender_criterion = nn.CrossEntropyLoss().cuda()
    age_criterion = nn.CrossEntropyLoss().cuda()

    for i in range(1, n_epoch+1):
        for j, (user_id, firstday, original_tensor) in enumerate(dataloader):
            loss = train(original_tensor, network, optimizer, gender_criterion, age_criterion)

        cur_lr = cur_lr * rate_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

    return loss


def extract_feature(encoder, datloader, max_length, feat_dim, outputfile="results/dummy.p"):

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


def val(eval, dataloader):

    gender_correct = 0
    age_correct = 0
    gender_count = 0
    age_count = 0
    for j, (feat, user_id, gender, age, firstday) in enumerate(dataloader):
        gender_guess, age_guess = eval(feat)

        _, predicted = torch.max(gender_guess.data, 1)
        gender_count += gender.size(0)
        gender_correct += (predicted == gender).sum()

        _, predicted = torch.max(age_guess.data, 1)
        age_count += age.size(0)
        age_correct += (predicted == age).sum()

    gender_correct = int(gender_correct) / int(gender_count) * 100
    age_correct = int(age_correct) / int(age_count) * 100

    return gender_correct, age_correct


def single_eval(file, message, nut, param, start, progress, isMLP, max_length):
    print(message)
    train_dataloader, val_dataloader, test_dataloader = tri_dataloader("data/subdata/classifier/" + file, param.batchSize)
    network = E2E(nut, param.featDim, isMLP, max_length)

    loss = trainEpochs(network, train_dataloader, param.epoch, learning_rate=param.lr, rate_decay=param.rateDecay)

    print(file, "\tFinal Loss: {0:.4f}\t".format(loss), timeSince(start, progress))
    if isMLP:
        outputfile = "results/classifier/" + file + "_MLP.p"
    else:
        outputfile = "results/classifier/" + file + "_direct.p"
    extract_feature(network, test_dataloader, day * part, param.featDim, outputfile=outputfile)

if __name__ == '__main__':
    parser = get_parser()
    param = parser.parse_args()
    start = time.time()

    parts = [1, 3, 6, 8]
    parts_sum = [0, 1, 4, 10]
    days = [1, 3, 7]
    days_sum = [1, 4, 11]
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(5)

    for i, part in enumerate(parts):
        for j, day in enumerate(days):
            progress = ((parts_sum[i] * 11 + days_sum[j] * parts[i]) * 2 - 1) / (11 * 18 * 2)
            filename = str(days) + "_days_" + str(parts) + "_parts_31"
            message = str(day) + " days, " + str(part) + "parts, direct"
            single_eval(filename, message, 31, param, start, progress, False, part*day)


            # MLP3層版も欲しい
            progress = (parts_sum[i] * 11 + days_sum[j] * parts[i]) / (11 * 18)
            filename = str(days) + "_days_" + str(parts) + "_parts_31"
            message = str(day) + " days, " + str(part) + "parts, MLP"
            single_eval(filename, message, 31, param, start, progress, True, part*day)
