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
    def __init__(self, data_file_path, csv_file_path, num_data=1, transform=None, loader=data_loader):
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
            data_route_info.append(information[:num_data])
            # transfer label from str to float
            labels_info.append([float(i) for i in information[num_data:len(information)]])
        self.num_data = num_data
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
        if self.num_data > 1:
            # read the data
            data = []
            for name in dataName:
                data_subset = self.loader(os.path.join(self.data_path, name))
                # parse the data
                if self.transform is not None:
                    data_subset = self.transform(data_subset)
                data.append(data_subset)  # data_subset = [c = 256, 256, 256]
            data = torch.cat(data, dim=0)  # data = [c = 256 * num_data, 256, 256]
        else:
            data = self.loader(os.path.join(self.data_path, dataName))
            if self.transform is not None:
                data = self.transform(data)
        # transfer to float tensor since loss in nn just accepts float
        label = torch.FloatTensor(label)
        return data, label

    # rewrite the function to show how much data points are in the data loader
    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        """
        # for achieving default_collate officially please refer to:
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
        """
