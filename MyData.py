import torch
from torch.utils.data import DataLoader
import pandas as pd


class GainDataset(torch.utils.data.Dataset):

    def __init__(self, file_name):
        gain_df = pd.read_csv(file_name)

        x = gain_df.iloc[:, 0:6].values
        y = gain_df.iloc[:, 6:9].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


