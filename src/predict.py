"""
Author : LiGorden
Email: likehao1006@gmail.com
"""
import os
import gc
import argparse

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from my_dataset import MyDataSet

from vit_model import SwinTransformerSys3D as create_model
from utils import evaluate


def predict(args):
    # load the data
    project_root = args.project_root  # get project root path
    data_path = os.path.join(project_root, "data", "data_model")  # data set path
    csv_path = os.path.join(project_root, "data", "data_route", args.csv_path)
    assert os.path.exists(data_path), "{} path does not exist.".format(data_path)
    assert os.path.exists(csv_path), "{} path does not exist.".format(csv_path)

    device = args.device

    data_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        ])

    # create model
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
                         drop_rate=0.,
                         attn_drop_rate=0.,
                         drop_path_rate=0.,
                         norm_layer=nn.LayerNorm,
                         patch_norm=True,
                         use_checkpoint=False,
                         frozen_stages=-1).to(device)

    # hyper param for testing
    model_weight_path = os.path.join(project_root, "result", "model_trained", args.weight_path, "checkpoint")
    for checkpoint in args.checkpoint_used:
        try:
            del state
        except:
            pass

        torch.cuda.empty_cache()
        print(gc.collect())

        state = torch.load(os.path.join(model_weight_path, checkpoint))

        model.load_state_dict(state["model_state_dict"])
        model.to(device)

        batch_size = args.batch_size

        if os.path.exists(os.path.join(args.save_path, checkpoint.split(".")[0])) is \
                False:
            os.makedirs(os.path.join(args.save_path, checkpoint.split(".")[0]))
        save_path = os.path.join(args.save_path, checkpoint.split(".")[0])

        if args.if_train_data:
            # create data set
            train_dataset = MyDataSet(data_file_path=os.path.join(data_path),
                                      csv_file_path=os.path.join(csv_path, "train",
                                                                 "data_model_train.csv"),
                                      num_data=args.num_data,
                                      weighted=args.weighted,
                                      transform=data_transform)

            train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            pin_memory=True,
                                                            num_workers=1,
                                                            )

            with torch.no_grad():
                torch.cuda.empty_cache()
                print(gc.collect())
                train_loss, predict_y_list, true_y_list = evaluate(model=model,
                                                                   data_loader=train_data_loader,
                                                                   device=device,
                                                                   epoch=0,
                                                                   aug_rot_90=args.aug_rot_90,
                                                                   weighted=args.weighted)
                np.save(file=os.path.join(save_path, "train_loss.npy"), arr=np.array(train_loss))
                np.save(file=os.path.join(save_path, "pred_train_data.npy"), arr=np.array(predict_y_list))
                np.save(file=os.path.join(save_path, "true_train_data.npy"), arr=np.array(true_y_list))

        if args.if_val_data:
            # create data set
            val_dataset = MyDataSet(data_file_path=os.path.join(data_path),
                                    csv_file_path=os.path.join(csv_path, "val",
                                                               "data_model_val.csv"),
                                    num_data=args.num_data,
                                    weighted=args.weighted,
                                    transform=data_transform)

            val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          num_workers=1,
                                                          )

            with torch.no_grad():
                torch.cuda.empty_cache()
                print(gc.collect())
                val_loss, predict_y_list, true_y_list = evaluate(model=model,
                                                                 data_loader=val_data_loader,
                                                                 device=device,
                                                                 epoch=0,
                                                                 aug_rot_90=args.aug_rot_90,
                                                                 weighted=args.weighted)
                np.save(file=os.path.join(save_path, "val_loss.npy"), arr=np.array(val_loss))
                np.save(file=os.path.join(save_path, "pred_val_data.npy"), arr=np.array(predict_y_list))
                np.save(file=os.path.join(save_path, "true_val_data.npy"), arr=np.array(true_y_list))

        if args.if_test_data:
            # create data set
            test_dataset = MyDataSet(data_file_path=os.path.join(data_path),
                                     csv_file_path=os.path.join(csv_path, "test",
                                                                "data_model_test.csv"),
                                     num_data=args.num_data,
                                     transform=data_transform)

            test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           pin_memory=True,
                                                           num_workers=1,
                                                           )

            with torch.no_grad():
                torch.cuda.empty_cache()
                print(gc.collect())
                test_loss, predict_y_list, true_y_list = evaluate(model=model,
                                                                  data_loader=test_data_loader,
                                                                  device=device,
                                                                  epoch=0,
                                                                  aug_rot_90=args.aug_rot_90)
                np.save(file=os.path.join(save_path, "test_loss.npy"), arr=np.array(test_loss))
                np.save(file=os.path.join(save_path, "pred_test_data.npy"), arr=np.array(predict_y_list))
                np.save(file=os.path.join(save_path, "true_test_data.npy"), arr=np.array(true_y_list))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    print(gc.collect())

    parser = argparse.ArgumentParser()
    # data_path is where the data and label are
    parser.add_argument('--project_root', type=str,
                        default=os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
    parser.add_argument('--csv_path', type=str, default="split_by_sample_pressure_token_all_features",
                        help='csv_path used for data_route')
    parser.add_argument('--if_train_data', type=bool, default=True)
    parser.add_argument('--if_val_data', type=bool, default=True)
    parser.add_argument('--if_test_data', type=bool, default=True)
    parser.add_argument('--num_data', type=int, default=7,
                        help='number of features used including token')
    parser.add_argument('--weighted', type=bool, default=False,
                        help='if the raw data include info about label weight')

    # model params
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--weight_path', type=str, default=os.path.join("ViT_3D_pressure_token_all_features"),
                        help='file path used for storing model weight')
    checkpoint_list = ["ckpt_best.pth", "ckpt_1.pth"] + \
                      ["ckpt_" + str(i) + ".pth" for i in range(5, 201, 5)]
    parser.add_argument('--checkpoint_used', type=list, default=checkpoint_list)

    parser.add_argument('--aug_rot_90', type=bool, default=False)
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

    # parser.add_argument('--checkpoint_used', type=list, default=["ckpt_15.pth"])
    parser.add_argument('--batch_size', type=int, default=4)

    # save path of checkout point
    parser.add_argument('--save_path', type=str,
                        default=os.path.abspath(os.path.join(os.getcwd(), "..", "..",
                                                             "result", "model_trained",
                                                             "ViT_3D_pressure_token_all_features")))

    opt = parser.parse_args()

    predict(opt)
