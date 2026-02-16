import numpy as np
import torch
from torch.utils.data import Dataset


class MultiStockDataset(Dataset):
    def __init__(self, data, stock_ids, lookback=30):

        self.X = []
        self.y = []
        self.stock_ids = []
        self.lookback = lookback

        for i in range(len(data) - lookback):
            self.X.append(data[i:i + lookback])
            self.y.append(data[i + lookback][-1])  # Returns column
            self.stock_ids.append(stock_ids[i + lookback])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        self.stock_ids = torch.tensor(np.array(self.stock_ids), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.stock_ids[idx], self.y[idx]
