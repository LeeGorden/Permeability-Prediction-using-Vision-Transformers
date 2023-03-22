import os

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
    def __init__(self, data_file_path, csv_file_path, transform=None, loader=data_loader):
        super(MyDataSet, self).__init__()
        # open the csv file that save the data route and labels
        fp = open(csv_file_path, 'r')
        fp.readline()
        data_route_info = []
        labels_info = []
        # store data and label
        for line in fp:
            line = line.strip('\n')
            information = line.split(',')
            data_route_info.append(information[0])
            # transfer label from str to float
            labels_info.append([float(i) for i in information[1:len(information)]])
        self.data_path = data_file_path
        self.data = data_route_info
        self.labels = labels_info
        self.transform = transform
        self.loader = loader

    # rewrite this function to customize reading data
    def __getitem__(self, item):
        # get the data name and data label
        dataName = self.data[item]
        label = self.labels[item]
        # read the data
        data = self.loader(os.path.join(self.data_path, dataName))
        # parse the data
        if self.transform is not None:
            data = self.transform(data)
        # transfer to float tensor since loss in nn just accepts float
        label = torch.FloatTensor(label)
        return data, label

    # rewrite the function to show how much data points are in the data loader
    def __len__(self):
        return len(self.data)
