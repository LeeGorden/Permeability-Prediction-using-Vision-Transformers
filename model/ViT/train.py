"""
Author : LiGorden
Email: likehao1006@gmail.com
"""
import os
import gc
from logzero import logger
import time
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_transform = {
        "train": transforms.Compose([
                                    transforms.ToTensor(),
                                    ]),
        "test": transforms.Compose([
                                   transforms.ToTensor(),
                                   ])}

    # load the data
    data_root = args.data_path  # get data root path
    data_path = os.path.join(data_root, "data_model")  # data set path
    csv_path = os.path.join(data_root, "data_route")
    assert os.path.exists(data_path), "{} path does not exist.".format(data_path)
    # create data set
    train_dataset = MyDataSet(data_file_path=os.path.join(data_path),
                              csv_file_path=os.path.join(csv_path, "split_by_sample", "train", "data_model_train.csv"),
                              transform=data_transform["train"])
    test_dataset = MyDataSet(data_file_path=os.path.join(data_path),
                             csv_file_path=os.path.join(csv_path, "split_by_sample", "test", "data_model_test.csv"),
                             transform=data_transform["test"])

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=1,
                                               )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=1,
                                              )

    model = create_model(img_size=256, num_classes=1, in_c=256, has_logits=False).to(device)

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
    optimizer = optim.Adam(pg, lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)

    best_error = np.inf
    best_epoch_time = [0, 0]
    save_path = args.save_path

    start_time = time.time()
    train_error = []
    test_error = []
    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch)

        logger.info(train_loss)
        train_error.append(float(train_loss))

        # test
        test_loss, predict_y_list, test_labels_list = evaluate(model=model,
                                                               data_loader=test_loader,
                                                               device=device,
                                                               epoch=epoch)

        logger.info(test_loss)
        test_error.append(float(test_loss))

        if test_loss < best_error:
            best_epoch_time[0] = round(time.time() - start_time, 2)
            best_epoch_time[1] = round((time.time() - start_time) / (epoch + 1), 2)
            best_error_pred_list = predict_y_list
            best_error_true_list = test_labels_list
            best_error = test_loss
            torch.save(model.state_dict(), os.path.join(save_path, "ViT-16.pth"))
            np.save(file=os.path.join(save_path, "pred.npy"), arr=np.array(best_error_pred_list))
            np.save(file=os.path.join(save_path, "true.npy"), arr=np.array(best_error_true_list))
            np.save(file=os.path.join(save_path, "time_best_epoch.npy"), arr=np.array(best_epoch_time))
            print('Model Saved')

    np.save(file=os.path.join(save_path, "train_loss.npy"), arr=np.array(train_error))
    np.save(file=os.path.join(save_path, "test_loss.npy"), arr=np.array(test_error))
    logger.info('Finished Training')
    logger.info('Best test mse: %.3f' % best_error)
    logger.info(best_epoch_time)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    print(gc.collect())

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # data_path is where the data and label are
    parser.add_argument('--data_path', type=str,
                        default=os.path.abspath(os.path.join(os.getcwd(), "../..", "data")))
    parser.add_argument('--save_path', type=str,
                        default=os.path.abspath(os.path.join(os.getcwd(), "../..", "result", "model_trained", "ViT")))
    parser.add_argument('--model-name', default='', help='create model name')

    # pre-trained weight, train from script if empty
    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')
    # check if need to freeze the layers
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

