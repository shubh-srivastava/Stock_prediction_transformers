import numpy as np
import torch
from torch.utils.data import Dataset


class MultiStockDataset(Dataset):
    def __init__(self, features, stock_ids, targets, sequence_length=30):

        self.X = []
        self.y = []
        self.stock_ids_out = [] 
        self.sequence_length = sequence_length

        unique_stock_ids = np.unique(stock_ids)

        for stock_id in unique_stock_ids:
            indices = np.where(stock_ids == stock_id)[0]
            stock_features = features[indices]
            stock_targets = targets[indices]

            if len(stock_features) <= self.sequence_length:
                continue

            for i in range(len(stock_features) - self.sequence_length):
                self.X.append(stock_features[i:i + self.sequence_length])
                self.y.append(stock_targets[i + self.sequence_length])
                self.stock_ids_out.append(stock_id)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        self.stock_ids = torch.tensor(np.array(self.stock_ids_out), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.stock_ids[idx], self.y[idx]
