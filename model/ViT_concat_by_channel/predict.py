"""
Author : LiGorden
Email: likehao1006@gmail.com
"""
import os
import gc

import numpy as np
import torch
from torchvision import transforms
from my_dataset import MyDataSet
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model
from utils import evaluate


# load the data
data_root = "C:\LiGorden\Research\poreflow\data"  # get data root path
data_path = os.path.join(data_root, "data_model")  # data set path
csv_path = os.path.join(data_root, "data_route")
assert os.path.exists(data_path), "{} path does not exist.".format(data_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
        # transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


# create data set
train_dataset = MyDataSet(data_file_path=os.path.join(data_path),
                          csv_file_path=os.path.join(csv_path, "split_by_sample_mpa_elec", "train",
                                                     "data_model_train.csv"),
                          num_data=2,
                          transform=data_transform)

test_dataset = MyDataSet(data_file_path=os.path.join(data_path),
                         csv_file_path=os.path.join(csv_path, "split_by_sample_mpa_elec", "test",
                                                    "data_model_test.csv"),
                         num_data=2,
                         transform=data_transform)

# create model
model = create_model(img_size=256, num_classes=1, in_c=256 * 2, has_logits=False).to(device)

# hyper param
model_weight_path = r"C:\LiGorden\Research\poreflow\result\model_trained\ViT_concat_by_channel"
state = torch.load(os.path.join(model_weight_path, "ViT-16.pth"))
model.load_state_dict(state)
model.to(device)
model.eval()
batch_size = 2
save_path = os.path.abspath(os.path.join(os.getcwd(), "../..", "result", "model_trained", "ViT_concat_by_channel"))

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=1,  # set to 1 for now
                                                # collate_fn=train_dataset.collate_fn
                                                )

test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=1,  # set to 1 for now
                                               # collate_fn=train_dataset.collate_fn
                                               )

with torch.no_grad():
    torch.cuda.empty_cache()
    print(gc.collect())
    test_loss, predict_y_list, true_y_list = evaluate(model=model,
                                                      data_loader=train_data_loader,
                                                      device=device,
                                                      epoch=1)
    np.save(file=os.path.join(save_path, "pred_train_data.npy"), arr=np.array(predict_y_list))
    np.save(file=os.path.join(save_path, "true_train_data.npy"), arr=np.array(true_y_list))

with torch.no_grad():
    torch.cuda.empty_cache()
    print(gc.collect())
    test_loss, predict_y_list, true_y_list = evaluate(model=model,
                                                      data_loader=test_data_loader,
                                                      device=device,
                                                      epoch=1)
    np.save(file=os.path.join(save_path, "pred_test_data.npy"), arr=np.array(predict_y_list))
    np.save(file=os.path.join(save_path, "true_test_data.npy"), arr=np.array(true_y_list))
