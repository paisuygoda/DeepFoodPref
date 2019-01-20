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

def split_dataloader(file, split_rate, batch_size):
    with open(file, mode='rb') as f:
        base = pickle.load(f)
    with open('data/subdata/user_attribute.p', mode='rb') as f:
        att = pickle.load(f)
    train_list = []
    val_list = []
    skipcount = 0
    for user_id, d_list in base.items():
        (gender, age_group, age, birthday) = att[user_id]

        if gender == 0 or age == 0:
            skipcount += 1
            continue
        gender -= 1
        age_group -= 1
        for (day, feat) in d_list:
            if np.random.rand() < split_rate:
                train_list.append((user_id, day, feat, gender, age))
            else:
                val_list.append((user_id, day, feat, gender, age))

    train_dataset = FoodPrefDataset(train_list)
    val_dataset = FoodPrefDataset(val_list)
    feat_size = len(train_dataset[0][0])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader, feat_size


class FoodPrefDataset(Dataset):
    def __init__(self, file):
        self.ds = file

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
        self.age = nn.Linear(input_size, 80)
        self.user = nn.Linear(input_size, 100)

    def forward(self, input):
        gender_guess = self.gender(input)
        age_guess = self.age(input)
        return gender_guess, age_guess


def train(feat, gender, age, eval, optimizer, gender_criterion, age_criterion):

    optimizer.zero_grad()

    gender_guess, age_guess = eval(feat)

    gender_loss = gender_criterion(gender_guess, gender)
    age_loss = age_criterion(age_guess, age)

    loss = gender_loss + age_loss

    loss.backward()
    optimizer.step()

    return loss.data.cpu().numpy()


def trainEpochs(eval, dataloader, n_epoch, learning_rate=0.01, rate_decay=0.9):
    start = time.time()
    cur_lr = learning_rate

    optimizer = optim.Adam(eval.parameters(), lr=learning_rate)
    gender_criterion = nn.CrossEntropyLoss().cuda()
    age_criterion = nn.CrossEntropyLoss().cuda()

    for i in range(1, n_epoch+1):
        for j, (feat, user_id, gender, age, firstday) in enumerate(dataloader):
            loss = train(feat, gender, age, eval, optimizer, gender_criterion, age_criterion)

        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
        cur_lr = cur_lr * rate_decay


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


def single_eval(feats, message):
    print(message)
    train_dataloader, val_dataloader, feat_size = split_dataloader(feats, 0.7, param.batchSize)
    eval = Classifier(feat_size)

    trainEpochs(eval, train_dataloader, param.epoch, learning_rate=param.lr, rate_decay=param.rateDecay)

    gender_accuracy, age_accuracy = val(eval, val_dataloader)
    print("Gender Accuracy:\t", gender_accuracy, "\nAge Accuracy:\t\t", age_accuracy)
    print("\n---\n")

if __name__ == '__main__':
    parser = get_parser()
    param = parser.parse_args()

    parts = [1, 3, 6, 8]
    days = [1, 3, 7]
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(5)

    for i, part in enumerate(parts):
        for j, day in enumerate(days):
            filename = "results/preffeat_LSTM_FM_" + str(day) + "_days_" + str(part) + "_parts_all_nut.p"
            message = str(day) + " days, " + str(part) + "parts, all"
            single_eval(filename, message)

            filename = "results/preffeat_LSTM_FM_" + str(day) + "_days_" + str(part) + "_parts_major_nut.p"
            message = str(day) + " days, " + str(part) + "parts, major"
            single_eval(filename, message)
