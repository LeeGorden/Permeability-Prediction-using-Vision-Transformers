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

from vit_model import vit_base_patch16_224_in21k as create_model
from utils import evaluate


# load the data
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))  # get data root path
data_path = os.path.join(project_root, "data", "data_model")  # data set path
csv_path = os.path.join(project_root, "data", "data_route")
assert os.path.exists(data_path), "{} path does not exist.".format(data_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    ])

# create data set
train_dataset = MyDataSet(data_file_path=os.path.join(data_path),
                          csv_file_path=os.path.join(csv_path, "split_by_sample", "train",
                                                     "data_model_train.csv"),
                          transform=data_transform)
test_dataset = MyDataSet(data_file_path=os.path.join(data_path),
                         csv_file_path=os.path.join(csv_path, "split_by_sample", "test",
                                                    "data_model_test.csv"),
                         transform=data_transform)

# create model
model = create_model(img_size=256, num_classes=1, in_c=256, has_logits=False).to(device)

# hyper param for testing
model_weight_path = os.path.join(project_root, "result", "model_trained", "ViT", "split_by_sample")
state = torch.load(os.path.join(model_weight_path, "ViT-16.pth"))
model.load_state_dict(state)
model.to(device)
model.eval()

batch_size = 2
save_path = os.path.join(project_root, "result", "model_trained", "ViT", "split_by_sample")

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=1,
                                                )

test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=1,
                                               )

with torch.no_grad():
    torch.cuda.empty_cache()
    print(gc.collect())
    test_loss, predict_y_list, true_y_list = evaluate(model=model,
                                                      data_loader=train_data_loader,
                                                      device=device,
                                                      epoch=0)
    np.save(file=os.path.join(save_path, "pred_train_data.npy"), arr=np.array(predict_y_list))
    np.save(file=os.path.join(save_path, "true_train_data.npy"), arr=np.array(true_y_list))

with torch.no_grad():
    torch.cuda.empty_cache()
    print(gc.collect())
    test_loss, predict_y_list, true_y_list = evaluate(model=model,
                                                      data_loader=test_data_loader,
                                                      device=device,
                                                      epoch=0)
    np.save(file=os.path.join(save_path, "pred_test_data.npy"), arr=np.array(predict_y_list))
    np.save(file=os.path.join(save_path, "true_test_data.npy"), arr=np.array(true_y_list))
