"""
Author : LiGorden
Email: likehao1006@gmail.com
"""
import os
from logzero import logger

import numpy as np
import torch


def data_loader(path):
    # numpy reader
    numpy_data = np.load(path)
    # transfer the numpy data info float numpy before transferred to tensor, other wise model will break down
    numpy_data = numpy_data.astype(np.float32)
    return numpy_data


# customize data reader
class MyDataSet(torch.nn.Module):
    def __init__(self, data_file_path, csv_file_path, num_data=1,
                 standardization=False, weighted=False,
                 transform=None, loader=data_loader):
        super(MyDataSet, self).__init__()
        # open the csv file that save the data route and labels
        fp = open(csv_file_path, 'r')
        fp.readline()
        data_route_info = []
        labels_info = []
        if weighted:
            weight_info = []
        # store data and label
        for line in fp:
            line = line.strip('\n')
            information = line.split(',')
            data_route_info.append(information[:num_data])
            # transfer label from str to float
            labels_info.append([float(i) for i in information[num_data:-1]])
            if weighted:
                weight_info.append([float(information[-1])])

        self.num_data = num_data
        self.data_path = data_file_path
        self.data = data_route_info
        self.labels = labels_info
        if weighted:
            self.weight = weight_info
        self.transform = transform
        self.weighted = weighted
        self.loader = loader

    # rewrite this function to customize reading data
    def __getitem__(self, item):
        # get the data name and data label
        dataName = self.data[item]
        label = self.labels[item]
        # read the data
        if self.num_data > 1:
            # read the data
            token = []
            data = []
            for name in dataName:
                if name.startswith("P_"):
                    if name == "P_1":
                        data_subset = torch.LongTensor([0])
                    elif name == "P_2":
                        data_subset = torch.LongTensor([1])
                    elif name == "P_5":
                        data_subset = torch.LongTensor([2])
                    elif name == "P_10":
                        data_subset = torch.LongTensor([3])
                    elif name == "P_20":
                        data_subset = torch.LongTensor([4])
                    token.append(data_subset)
                else:
                    data_subset = self.loader(os.path.join(self.data_path, name))
                    # parse the data
                    if self.transform is not None:
                        data_subset = self.transform(data_subset)
                        data_subset = data_subset.permute(2, 0, 1)  # make data from (L, W, D) to (D, L, W)
                        data_subset = data_subset.unsqueeze_(0)  # (D, L, W) -> (1, D, L, W), 1 is dim_num
                    data.append(data_subset)
            data = [torch.cat(data, dim=0)]  # (1, D, L, W) -> [(Channel_num(feature_num), D, L, W)]
            data = token + data  # [token, (Channel_num(feature_num), D, L, W)]
        else:
            data = self.loader(os.path.join(self.data_path, dataName))
            if self.transform is not None:
                data = self.transform(data)
                data = data.permute(2, 0, 1)
                data = data.unsqueeze(0)  # now treat the 3D array as 3D object. Thus create channel at first dimension
        # transfer to float tensor since loss in nn just accepts float
        label = torch.FloatTensor(label)
        if self.weighted:
            weight = self.weight[item]
            weight = torch.FloatTensor(weight)
            return data, label, weight
        return data, label

    # rewrite the function to show how much data points are in the data loader
    def __len__(self):
        return len(self.data)
