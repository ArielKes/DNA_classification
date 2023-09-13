import os

import h5py
import numpy as np
import torch

from dataset import CovidDataset


# 1D dataset
class CovidDataset1D(CovidDataset):
    def __init__(self, data_dir, transform=None):
        super().__init__(data_dir, transform)

    def __getitem__(self, index):
        with h5py.File(os.path.join(self.data_dir, 'data.h5'), 'r') as f:
            seq_dataset = f['data']['sequences']
            label_dataset = f['data']['labels']
            seq, label = seq_dataset[index], label_dataset[index]
        row_indices = np.argmax(seq, axis=1)

        row_indices = torch.tensor(row_indices, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return row_indices, label
