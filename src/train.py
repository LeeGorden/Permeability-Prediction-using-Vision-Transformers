"""
Author : LiGorden
Email: likehao1006@gmail.com
"""
import os
import gc
from logzero import logger
import time
import argparse
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from src.utils.my_dataset import MyDataSet
from src.model.vit_model import SwinTransformerSys3D as create_model
from src.utils.utils import get_distribution, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(device)

    data_transform = {
        "train": transforms.Compose([
                                    transforms.ToTensor(),
                                    ]),
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   ])}

    # load the data
    data_root = args.data_path  # get data root path
    data_path = os.path.join(data_root, "data_model")  # data set path
    csv_path = os.path.join(data_root, "data_route", args.csv_path)
    assert os.path.exists(data_path), "{} path does not exist.".format(data_path)

    # split train and val data
    if os.path.exists(os.path.join(csv_path, "train")) is False:
        os.makedirs(os.path.join(csv_path, "train"))
    if os.path.exists(os.path.join(csv_path, "val")) is False:
        os.makedirs(os.path.join(csv_path, "val"))

    if args.split_train_val:
        csv_train_val = pd.read_csv(os.path.join(csv_path, "train_val", "data_model_train_val.csv"))
        columns_name = list(csv_train_val.columns)
        csv_train_val = np.array(csv_train_val)
        X, y = csv_train_val[:, :-1].reshape(-1, len(columns_name) - 1), csv_train_val[:, -1].reshape(-1, 1)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_ratio, random_state=0)
        data_train = np.concatenate([X_train, y_train], axis=1)
        data_train = pd.DataFrame(data=data_train, columns=columns_name, index=None)
        data_train.to_csv(os.path.join(csv_path, "train", "data_model_train.csv"), index=None)
        data_val = np.concatenate([X_val, y_val], axis=1)
        data_val = pd.DataFrame(data=data_val, columns=columns_name, index=None)
        data_val.to_csv(os.path.join(csv_path, "val", "data_model_val.csv"), index=None)

    # create data set
    train_dataset = MyDataSet(data_file_path=data_path,
                              csv_file_path=os.path.join(csv_path, "train", "data_model_train.csv"),
                              num_data=args.num_data,
                              weighted=args.weighted,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(data_file_path=data_path,
                            csv_file_path=os.path.join(csv_path, "val", "data_model_val.csv"),
                            num_data=args.num_data,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               # num_workers=0,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             # num_workers=0,
                                             )

    train_distribution = None

    # get distribution(mean, std) of train data
    if args.standardization_file_name != "None":
        if os.path.exists(os.path.join(csv_path, "train", args.standardization_file_name)) is False:
            logger.info("calculating train_data distribution")
            # train_distribution -> numpy(dim, 2) = numpy(dim, (mean, std))
            train_distribution = get_distribution(data_loader=train_loader, device=device, weighted=args.weighted)
            np.save(file=os.path.join(csv_path, "train", args.standardization_file_name), arr=train_distribution)
            train_distribution = torch.FloatTensor(train_distribution)
        else:
            logger.info("loading train_data distribution")
            train_distribution = np.load(os.path.join(csv_path, "train", args.standardization_file_name))
            train_distribution = torch.FloatTensor(train_distribution)

    model = create_model(pretrained=None,
                         pretrained2d=False,
                         patch_size=args.patch_size,
                         in_c=args.in_c,
                         num_classes=args.num_classes,
                         embed_dim=args.embed_dim,
                         depths=args.depths,
                         num_heads=args.num_heads,
                         window_size=args.window_size,
                         mlp_ratio=4.,
                         qkv_bias=True,
                         qk_scale=None,
                         drop_rate=args.drop_rate,
                         attn_drop_rate=args.attn_drop_rate,
                         drop_path_rate=args.drop_path_rate,
                         norm_layer=nn.LayerNorm,
                         patch_norm=True,
                         use_checkpoint=False,
                         init_weight=args.init_weight,
                         frozen_stages=-1).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # if freeze everything except for head, pre_logits
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "Adam":
        optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(pg, lr=args.lr, weight_decay=args.weight_decay)

    if args.use_lr_schedule:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=args.patience)

    best_loss = np.inf
    best_epoch_time = [0, 0]

    save_path = args.save_path
    if os.path.exists(os.path.join(save_path, "checkpoint")) is False:
        os.makedirs(os.path.join(save_path, "checkpoint"))

    if args.resume:
        path_checkpoint = os.path.join(save_path, "checkpoint", args.resume_file)
        checkpoint = torch.load(path_checkpoint)  # load checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # load learnable params in model
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['last_epoch']
        es_status = checkpoint['es'] if args.use_es_input else 0
        best_loss = checkpoint['best_loss']
        train_error = checkpoint['train_error']
        val_error = checkpoint['val_error']
        if args.use_lr_schedule:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        last_epoch = -1
        train_error = []
        val_error = []

    start_time = time.time()
    # es = es_status if args.resume else 0
    es = es_status if args.resume and es_status < args.es else 0
    for epoch in range(last_epoch + 1, args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     standardization=train_distribution,
                                     epoch=epoch,
                                     aug_rot_90=args.aug_rot_90,
                                     weighted=args.weighted)

        # logger.info(train_loss)
        train_error.append(float(train_loss))

        # val
        val_loss, _, _ = evaluate(model=model,
                                  data_loader=val_loader,
                                  device=device,
                                  standardization=train_distribution,
                                  epoch=epoch)
        if args.use_lr_schedule:
            scheduler.step(val_loss)
        # logger.info(val_loss)
        val_error.append(float(val_loss))

        # save checkpoint
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "last_epoch": epoch,
                      "es": args.es,
                      "best_loss": val_loss if val_loss < best_loss else best_loss,
                      "train_error": train_error,
                      "val_error": val_error}
        if args.use_lr_schedule:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if (epoch + 1) % args.autosave_period == 0 or epoch == 0:
            torch.save(checkpoint, os.path.join(save_path, "checkpoint",
                                                "ckpt_%s.pth" % (str(epoch + 1))))

        if val_loss < best_loss:
            es = 1
            best_epoch_time[0] = round(time.time() - start_time, 2)
            best_epoch_time[1] = round((time.time() - start_time) / (epoch + 1), 2)
            best_loss = val_loss
            torch.save(checkpoint, os.path.join(save_path, "checkpoint",
                                                "ckpt_best.pth"))
            np.save(file=os.path.join(save_path, "time_best_epoch.npy"), arr=np.array(best_epoch_time))
            print('Model Saved')
        else:
            es += 1
            print("Counter {} of %.0f".format(es) % args.es)

            if es > args.es:
                print("Early stopping with best_loss: ", best_loss, "and val_acc for this epoch: ", val_loss)
                break

    np.save(file=os.path.join(save_path, "train_loss.npy"), arr=np.array(train_error))
    np.save(file=os.path.join(save_path, "val_loss.npy"), arr=np.array(val_error))
    # logger.info('Finished Training')
    # logger.info('Best test mse: %.3f' % best_loss)
    # logger.info(best_epoch_time)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    print(gc.collect())

    parser = argparse.ArgumentParser()
    # data_path is where the data and label are
    parser.add_argument('--data_path', type=str,
                        default=os.path.abspath(os.path.join(os.getcwd(), "../..", "data")))
    parser.add_argument('--csv_path', type=str, default="split_by_sample_pressure_token_all_features",
                        help='csv_path used for data_route')
    parser.add_argument('--split_train_val', type=bool, default=False,
                        help='split train val or not')
    parser.add_argument('--val_ratio', type=float, default=0.,
                        help='ratio of validation data')
    parser.add_argument('--random_state', type=float, default=0,
                        help='random seed of splitting train and val data')
    parser.add_argument('--num_data', type=int, default=7,
                        help='number of features used including token')
    parser.add_argument('--standardization_file_name', type=str, default="None",
                        help='csv_path used for standardization of each input dimension of 3D')
    parser.add_argument('--weighted', type=bool, default=False,
                        help='if the raw data include info about label weight')

    # model params
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument('--patch_size', type=tuple, default=(8, 8, 8))
    parser.add_argument('--in_c', type=int, default=6)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=96,
                        help='dim of first SwinTransformer Layer')
    parser.add_argument('--depths', type=list, default=[2, 2, 2, 2],
                        help='depth of each SwinTransformer Layer')
    parser.add_argument('--num_heads', type=list, default=[8, 16, 32, 32],
                        help='number of attention head in each SwinTransformer Layer')
    parser.add_argument('--window_size', type=tuple, default=(4, 4, 4),
                        help='size of shift window')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='dropout rate of projection in attention structure, including mlp and pos_drop'
                        )
    parser.add_argument('--attn_drop_rate', type=float, default=0.00,
                        help='dropout rate of attention structure')
    parser.add_argument('--drop_path_rate', type=float, default=0.00,
                        help='dropout rate of residual path')

    parser.add_argument('--aug_rot_90', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--lr', type=float, default=1e-5)  # 1e-5
    parser.add_argument('--use_lr_schedule', type=bool, default=True,
                        help='use lr_schedule')
    parser.add_argument('--patience', type=float, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 1e-4
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dampening', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=False)

    # pre-trained weight/initialization weight, train from script if empty
    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')
    parser.add_argument('--init_weight', type=str, default="Constant",
                        help='way of initializing weights')
    parser.add_argument('--es', type=int, default=10,
                        help='early stopping threshold')

    # check if need to freeze the layers
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # check if resume training from checkpoint
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume training from checkpoint')
    parser.add_argument('--resume_file', type=str, default="ckpt_best.pth",
                        help='file used to resume training from checkpoint')
    parser.add_argument('--use_es_input', type=bool, default=True,
                        help='use the early stopping status input or not')
    # save path of checkout point
    parser.add_argument('--autosave_period', type=int, default=5,
                        help='save of model happen after the number of epoch set')
    parser.add_argument('--save_path', type=str,
                        default=os.path.abspath(os.path.join(os.getcwd(), "../..", "result", "model_trained"
                                                             )))

    opt = parser.parse_args()

    main(opt)
