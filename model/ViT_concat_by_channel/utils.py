import os
from logzero import logger
import sys
import json
import pickle
import random

import numpy as np

import torch
from torch import nn
from tqdm import tqdm

"""
def calculate_prediction(model_predicted, accuracy_th=0.5):
    # Todo: make it MSE loss or we could delete if if not necessary
    predicted_result = model_predicted > accuracy_th  # tensor之间比较大小跟array一样
    predicted_result = predicted_result.float()  # 因为label是0, 1的float, 所以这里需要将他们转化为同类型tensor

    return predicted_result
"""


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    logger.info(f"Epoch {epoch + 1}")
    model.train()
    loss_function = nn.MSELoss()
    # train_loss is the accumulate loss across steps in each epoch
    train_loss = 0.0
    # train_acc_all_labels is used to calculate the avg performance of all labels
    # train_acc_all_labels = 0.0
    # train_acc_separate_labels is used to calculate the performance of all labels separately
    # total number of labels = len(train_loader.dataset.labels[0]) labels
    # train_acc_separate_labels = torch.tensor([0 for _ in range(len(data_loader.dataset.labels[0]))]).float()
    # train_acc_separate_labels = train_acc_separate_labels.to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = model(images)

        # train_acc_all_labels += abs((pred - labels).sum().item()) / labels.size()[1]
        # train_acc_separate_labels += torch.eq(pred, labels).float().sum(dim=0)

        loss = loss_function(pred, labels)
        loss.backward()
        train_loss += loss.item()

        data_loader.desc = "[train epoch {}] mse: {:.3f}".\
            format(epoch + 1,
                   train_loss / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return train_loss / (step + 1)  # train_acc_all_labels / sample_num  train_acc_separate_labels / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = nn.MSELoss()

    model.eval()

    test_loss = 0.0
    # test_acc_all_labels = 0.0
    # test_acc_separate_labels = torch.tensor([0 for _ in range(len(data_loader.dataset.labels[0]))]).float()
    # test_acc_separate_labels = test_acc_separate_labels.to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)
    predict_y_list = list()
    test_labels_list = list()
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = model(images)
        # test_acc_all_labels += (pred - labels).sum().item() / labels.size()[1]
        # test_acc_separate_labels += torch.eq(pred_class, labels).float().sum(dim=0)
        predict_y_list.extend(list(np.array(pred.cpu())))
        test_labels_list.extend(list(np.array(labels.cpu())))

        loss = loss_function(pred, labels)
        test_loss += loss

        data_loader.desc = "[test epoch {}] mse: {:.3f}".\
            format(epoch + 1,
                   test_loss / (step + 1))

    return test_loss / (step + 1), predict_y_list, test_labels_list
