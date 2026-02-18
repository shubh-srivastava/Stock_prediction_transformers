import numpy as np
import torch
from torch.utils.data import Dataset


class MultiStockDataset(Dataset):
    def __init__(self, features, stock_ids, targets, sequence_length=30):

        self.X = []
        self.y = []
        self.stock_ids_out = []  # Use a different name to avoid confusion
        self.sequence_length = sequence_length

        # Find the unique stock IDs present in the data
        unique_stock_ids = np.unique(stock_ids)

        for stock_id in unique_stock_ids:
            # Find the indices in the original arrays that correspond to the current stock
            indices = np.where(stock_ids == stock_id)[0]

            # Extract the data for this single stock
            stock_features = features[indices]
            stock_targets = targets[indices]

            # Ensure there's enough data to create at least one sequence
            if len(stock_features) <= self.sequence_length:
                continue

            # Create sliding windows for this stock
            for i in range(len(stock_features) - self.sequence_length):
                # The input sequence is a window of features
                self.X.append(stock_features[i:i + self.sequence_length])
                # The target is the return on the day AFTER the window ends
                self.y.append(stock_targets[i + self.sequence_length])
                # The stock_id for this sequence is the one we're processing
                self.stock_ids_out.append(stock_id)

        # Convert lists to tensors
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        self.stock_ids = torch.tensor(np.array(self.stock_ids_out), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.stock_ids[idx], self.y[idx]
