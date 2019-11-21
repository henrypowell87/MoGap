import torch
import numpy as np
from torch.utils import data


class DataSet(data.Dataset):
    def __init__(self, list_IDS, ground_truths, data_dir, transform=None):
        self.ground_truths = ground_truths
        self.list_IDS = list_IDS
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.list_IDS)

    def __getitem__(self, index):
        ID = self.list_IDS[index]
        x = np.load(self.data_dir + ID)
        if self.transform:
            x = self.transform(x)
        y = self.ground_truths[ID]

        return x, y

