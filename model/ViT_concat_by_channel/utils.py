"""
Author : LiGorden
Email: likehao1006@gmail.com
"""
from logzero import logger
import sys

import numpy as np

import torch
from torch import nn
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    logger.info(f"Epoch {epoch + 1}")
    model.train()
    loss_function = nn.MSELoss()
    train_loss = 0.0
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = model(images)
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

    return train_loss / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = nn.MSELoss()

    model.eval()

    test_loss = 0.0
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
        predict_y_list.extend(list(np.array(pred.cpu())))
        test_labels_list.extend(list(np.array(labels.cpu())))

        loss = loss_function(pred, labels)
        test_loss += loss

        data_loader.desc = "[test epoch {}] mse: {:.3f}".\
            format(epoch + 1,
                   test_loss / (step + 1))

    return test_loss / (step + 1), predict_y_list, test_labels_list
