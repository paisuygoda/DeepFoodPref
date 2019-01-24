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
    parser.add_argument('--featDim', default=31, type=int)
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
        if type(user_id) == int:
            user_id = str(user_id)
        (gender, age_group, age, birthday) = att[user_id]

        if gender == 0 or age == 0:
            skipcount += 1
            continue
        gender -= 1
        age_group -= 1
        for (day, feat) in d_list:
            if np.random.rand() < split_rate:
                train_list.append((user_id, day, torch.FloatTensor(feat), gender, age))
            else:
                val_list.append((user_id, day, torch.FloatTensor(feat), gender, age))

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
        age = self.ds[idx][4] / 100
        return feat, user_id, gender, age, firstday


class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.gender = nn.Linear(input_size, 2)
        self.age = nn.Linear(input_size, 1)
        self.user = nn.Linear(input_size, 100)

    def forward(self, input):
        gender_guess = self.gender(input)
        age_guess = self.age(input)
        return gender_guess, age_guess


def train(feat, gender, age, eval, optimizer, gender_criterion, age_criterion):

    optimizer.zero_grad()

    gender_guess, age_guess = eval(feat)

    gender_loss = gender_criterion(gender_guess, gender)
    age_loss = age_criterion(age_guess, age.float())

    loss = gender_loss + age_loss

    loss.backward()
    optimizer.step()

    return loss.data.cpu().numpy()


def trainEpochs(eval, dataloader, n_epoch, learning_rate=0.01, rate_decay=0.9):
    cur_lr = learning_rate

    optimizer = optim.Adam(eval.parameters(), lr=learning_rate)
    gender_criterion = nn.CrossEntropyLoss().cuda()
    age_criterion = nn.MSELoss().cuda()

    for i in range(1, n_epoch+1):
        for j, (feat, user_id, gender, age, firstday) in enumerate(dataloader):
            train(feat, gender, age.float().view(age.size()[0], 1), eval, optimizer, gender_criterion, age_criterion)

        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
        cur_lr = cur_lr * rate_decay


def val(eval, dataloader):

    gender_correct = 0
    age_correct = 0
    gender_count = 0
    age_count = 0
    mseloss = nn.MSELoss().cuda()
    for j, (feat, user_id, gender, age, firstday) in enumerate(dataloader):
        gender_guess, age_guess = eval(feat)

        _, predicted = torch.max(gender_guess.data, 1)
        gender_count += gender.size(0)
        gender_correct += (predicted == gender).sum()

        age_count += 1
        loss = mseloss(age_guess.data, age.float().view(age.size()[0], 1))
        age_correct += loss.data.cpu().numpy()

    gender_correct = int(gender_correct) / int(gender_count) * 100
    age_correct = math.sqrt(float(age_correct) / age_count) * 100

    return gender_correct, age_correct


def single_eval(feats, message):
    print(message)
    train_dataloader, val_dataloader, feat_size = split_dataloader(feats, 0.7, param.batchSize)
    eval = Classifier(feat_size)

    trainEpochs(eval, train_dataloader, param.epoch, learning_rate=param.lr, rate_decay=param.rateDecay)

    gender_accuracy, age_accuracy = val(eval, val_dataloader)
    print("Gender Accuracy:\t", gender_accuracy, "\nAge Accuracy:\t\t", age_accuracy)
    print("\n---\n")
    with open("results/bof.csv", "a") as f:
        f.write(message + "," + str(round(gender_accuracy,2)))
        f.write(message + "," + str(round(age_accuracy, 2)))

def make_dataset_forcemeals():
    with open('data/subdata/user_daily_meals.p', mode='rb') as f:
        user_daily_meals = pickle.load(f)
    final_dataset = {}
    nut_num = 31
    for user_id, daily_meals in user_daily_meals.items():
        meals_list = []
        for day, meals in sorted(daily_meals.items()):
            meals_list.append(np.asarray(np.zeros(nut_num)))
            for time, meal in meals:
                for i in range(len(meals_list)):
                    meals_list[i] += np.asarray(meal)
            if len(meals_list) == 7:
                if user_id in final_dataset:
                    final_dataset[user_id].append((day, torch.FloatTensor(meals_list[0] / 7)))
                else:
                    final_dataset[user_id] = [(day, torch.FloatTensor(meals_list[0] / 7))]
                meals_list = meals_list[1:]

    with open("data/subdata/user_meals_dataset_baseline_7day_1part_average.p", "wb") as f:
        pickle.dump(final_dataset, f)

if __name__ == '__main__':
    parser = get_parser()
    param = parser.parse_args()

    make_dataset_forcemeals()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(5)

    # filename = "data/subdata/user_meals_dataset_baseline_7day_1part_average.p"
    # message = "baseline - only sum"
    # single_eval(filename, message)

    dims = [2,5,10,20,30,50,100,200]
    terms = [1,7,14,30,60]
    for i, part in dims:
        filename = "data/subdata/user_vec_by_bow_" + str(i) + "dim_30_day.p"
        message = "30 days " + str(i) + "dim"
        single_eval(filename, message)

    for j, day in terms:
        filename = "data/subdata/user_vec_by_bow_20dim_" + str(j) + "_day.p"
        message = str(j) + " days 20 dim"
        single_eval(filename, message)

