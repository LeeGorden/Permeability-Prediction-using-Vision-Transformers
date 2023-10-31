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
import torch


def get_distribution(data_loader, device, weighted=False):
    """
    function: get the mean and std of each dim of input across Sample_size * D * L * W
    return : np.array([tensor_mean(dim), tensor_std(dim)])
    """
    data_loader = tqdm(data_loader)
    data = list()
    for step, sub_data in enumerate(data_loader):
        if weighted:
            images, _, _ = sub_data
        else:
            images, _ = sub_data
        images_content = images[1]  # (Batch_size, dim, D, L, W)
        images_content = images_content.to(device)
        data.append(images_content)
    data = torch.cat(data, dim=0)  # (Sample_size, dim, D, L, W)
    data = data.permute(1, 0, 2, 3, 4)  # (dim, Sample_size, D, L, W)
    data = torch.flatten(data, start_dim=1)  # (dim, Sample_size * D * L * W)
    result = np.concatenate([np.array(torch.mean(data, dim=1)).reshape(1, -1).to(device),
                             np.array(torch.std(data, dim=1)).reshape(1, -1).to(device)],
                            axis=0)

    return result


def train_one_epoch(model, optimizer, data_loader, device, epoch, standardization=None, aug_rot_90=False, weighted=False):
    logger.info(f"Epoch {epoch + 1}")
    model.train()
    loss_function = nn.MSELoss(reduction='none')
    train_loss = 0.0

    sample_num = 0
    data_loader = tqdm(data_loader)

    if standardization is not None:
        mean, std = standardization

    for step, data in enumerate(data_loader):
        if weighted:
            images, labels, weights = data
            weights = weights.to(device)
        else:
            images, labels = data
        images_token = images[0]
        images_content = images[1]  # (B, C, D, L, W)

        if standardization is not None:
            # conduct standardization on images dim(C) by dim(C) across B, C, D, L, W
            images_content = images_content.permute(0, 2, 3, 4, 1)  # (B, D, L, W, C)
            B, D, L, W, C = images_content.size()
            images_content = torch.flatten(images_content, start_dim=0, end_dim=4)  # (B * D * L * W, C)
            images_content = torch.div(torch.sub(images_content, mean), std)  # (B * D * L * W, C)
            images_content = images_content.view(B, D, L, W, C).permute(0, 4, 1, 2, 3)  # (B, C, D, L, W)
            logger.info(torch.mean(torch.flatten(images_content.permute(0, 2, 3, 4, 1), start_dim=0, end_dim=4), dim=0))

        sample_num += images_content.shape[0]
        images_token = images_token.to(device)
        images_content = images_content.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model([images_token, images_content])
        loss = loss_function(pred, labels)
        loss = torch.mul(loss, weights) if weighted else loss
        train_loss += torch.sum(loss)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        if aug_rot_90:
            for rot_time in [1, 2, 3]:
                # dims = (Batch_size, dim, D, L, W)
                images_aug = torch.rot90(images_content, k=rot_time, dims=[3, 4])  # rotate between Length and Width
                images_token = images_token.to(device)
                images_aug = images_aug.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                pred = model([images_token, images_aug])
                loss = loss_function(pred, labels)
                loss = torch.mul(loss, weights) if weighted else loss
                train_loss += float(torch.sum(loss))
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                sample_num += images_aug.shape[0]

                """
                # elif aug_status == 1:
                images_aug = torch.rot90(images_content, k=rot_time, dims=[2, 3])
                images_token = images_token.to(device)
                images_aug = images_aug.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                pred = model([images_token, images_aug])
                loss = loss_function(pred, labels)
                loss = torch.mul(loss, weights) if weighted else loss
                train_loss += float(torch.sum(loss))
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                sample_num += images_aug.shape[0]

                # elif aug_status == 2:
                images_aug = torch.rot90(images_content, k=rot_time, dims=[2, 4])
                images_token = images_token.to(device)
                images_aug = images_aug.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                pred = model([images_token, images_aug])
                loss = loss_function(pred, labels)
                loss = torch.mul(loss, weights) if weighted else loss
                train_loss += float(torch.sum(loss))
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                sample_num += images_aug.shape[0]
                """

        data_loader.desc = "[train epoch {}] mse: {:.3f}".\
            format(epoch + 1,
                   train_loss / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return train_loss / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, standardization=None, aug_rot_90=False, weighted=False):
    model.eval()
    loss_function = nn.MSELoss(reduction='none')

    test_loss = 0.0
    sample_num = 0
    data_loader = tqdm(data_loader)
    if standardization is not None:
        mean, std = standardization
    predict_y_list = list()
    test_labels_list = list()
    for step, data in enumerate(data_loader):
        if weighted:
            images, labels, weights = data
            weights = weights.to(device)
        else:
            images, labels = data
        # logger.info(labels)
        images_token = images[0]
        images_content = images[1]

        if standardization is not None:
            # conduct standardization on images dim(C) by dim(C) across B, C, D, L, W
            images_content = images_content.permute(0, 2, 3, 4, 1)  # (B, D, L, W, C)
            B, D, L, W, C = images_content.size()
            images_content = torch.flatten(images_content, start_dim=0, end_dim=4)  # (B * D * L * W, C)
            images_content = torch.div(torch.sub(images_content, mean), std)  # (B * D * L * W, C)
            images_content = images_content.view(B, D, L, W, C).permute(0, 4, 1, 2, 3)  # (B, C, D, L, W)
            logger.info(torch.mean(torch.flatten(images_content.permute(0, 2, 3, 4, 1), start_dim=0, end_dim=4), dim=0))

        sample_num += images_content.shape[0]
        images_token = images_token.to(device)
        images_content = images_content.to(device)
        labels = labels.to(device)

        pred = model([images_token, images_content])
        predict_y_list.extend(list(np.array(pred.cpu())))
        test_labels_list.extend(list(np.array(labels.cpu())))
        loss = loss_function(pred, labels)
        loss = torch.mul(loss, weights) if weighted else loss
        test_loss += float(torch.sum(loss))

        if aug_rot_90:
            for rot_time in [1, 2, 3]:

                images_aug = torch.rot90(images_content, k=rot_time, dims=[3, 4])
                images_aug = images_aug.to(device)
                pred = model([images_token, images_aug])
                predict_y_list.extend(list(np.array(pred.cpu())))
                test_labels_list.extend(list(np.array(labels.cpu())))
                loss = loss_function(pred, labels)
                loss = torch.mul(loss, weights) if weighted else loss
                test_loss += float(torch.sum(loss))
                sample_num += images_aug.shape[0]

                images_aug = torch.rot90(images_content, k=rot_time, dims=[2, 3])
                images_aug = images_aug.to(device)
                pred = model([images_token, images_aug])
                predict_y_list.extend(list(np.array(pred.cpu())))
                test_labels_list.extend(list(np.array(labels.cpu())))
                loss = loss_function(pred, labels)
                loss = torch.mul(loss, weights) if weighted else loss
                test_loss += float(torch.sum(loss))
                sample_num += images_aug.shape[0]
                
                images_aug = torch.rot90(images_content, k=rot_time, dims=[2, 4])
                images_aug = images_aug.to(device)
                pred = model([images_token, images_aug])
                predict_y_list.extend(list(np.array(pred.cpu())))
                test_labels_list.extend(list(np.array(labels.cpu())))
                loss = loss_function(pred, labels)
                loss = torch.mul(loss, weights) if weighted else loss
                test_loss += float(torch.sum(loss))
                sample_num += images_aug.shape[0]

        data_loader.desc = "[test epoch {}] mse: {:.3f}".\
            format(epoch + 1,
                   test_loss / sample_num)

    return test_loss / sample_num, predict_y_list, test_labels_list
