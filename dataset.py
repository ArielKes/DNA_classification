import re
import os
import gc
import h5py
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CovidDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(CovidDataset).__init__()
        self.data_dir = data_dir
        self.base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.nucleotides = re.compile('[^ATCG]')
        self.transform = transform

        if not os.path.exists(os.path.join(self.data_dir, "data.h5")):
            self.cache_data_in_batches()

    def cache_data(self):
        sequences = []
        labels = []
        for label in os.listdir(self.data_dir):
            print(label)
            fasta_file = os.path.join(self.data_dir, label, "sequences.fasta")
            with open(fasta_file, "r") as f:
                for record in tqdm(SeqIO.parse(f, "fasta")):
                    cleaned = self.nucleotides.sub('', str(record.seq))
                    # Convert nucleotide sequence to an array of integers
                    # integer_seq = np.array([self.base_map[base] for base in cleaned])
                    integer_seq = torch.tensor([self.base_map[base] for base in cleaned], dtype=torch.long)
                    # Perform one-hot encoding of the integer sequence
                    one_hot_seq = torch.zeros(len(integer_seq), len(self.base_map), dtype=torch.float)
                    one_hot_seq.scatter_(1, integer_seq.unsqueeze(1), 1)
                    # one_hot_seq = sp.csr_matrix(([1] * len(integer_seq), (range(len(integer_seq)), integer_seq)),
                    #                             shape=(len(integer_seq), len(self.base_map)))

                    sequences.append(one_hot_seq)
                    labels.append(label)
        print("finished reading, starting to process the data and save a cached h5 file")
        self.string_to_int = {string: i for i, string in enumerate(set(labels))}
        labels = [self.string_to_int[string] for string in labels]
        # Pad sequences to the same length
        sequences = pad_sequence([seq for seq in sequences], batch_first=True)
        labels = torch.tensor(labels, dtype=torch.long)
        with h5py.File(os.path.join(self.data_dir, "data.h5"), 'w') as f:
            grp = f.create_group('data')

            # Create a dataset for your sequences and save it to the group as a table
            seq_dataset = grp.create_dataset('sequences', data=sequences, maxshape=(None,))
            seq_dataset.attrs.create('field_0_name', 'index')
            seq_dataset.attrs.create('field_1_name', 'sequence')

            # Create a dataset for your labels and save it to the group as a table
            label_dataset = grp.create_dataset('labels', data=labels, maxshape=(None,))
            label_dataset.attrs.create('field_0_name', 'index')
            label_dataset.attrs.create('field_1_name', 'label')

    def cache_data_in_batches(self):
        sequences = []
        labels = []

        # Process and save sequences in batches
        batch_size = 1000  # Adjust the batch size according to available memory
        label_int = -1
        for label in os.listdir(self.data_dir):
            print(label)
            fasta_file = os.path.join(self.data_dir, label, "sequences.fasta")
            if not os.path.exists(fasta_file):
                continue
            label_int += 1
            with open(fasta_file, "r") as f:
                for i, record in tqdm(enumerate(SeqIO.parse(f, "fasta"))):
                    cleaned = self.nucleotides.sub('', str(record.seq))
                    # Convert nucleotide sequence to an array of integers
                    integer_seq = torch.tensor([self.base_map[base] for base in cleaned], dtype=torch.long)
                    # Perform one-hot encoding of the integer sequence
                    one_hot_seq = torch.zeros(len(integer_seq), len(self.base_map), dtype=torch.float)
                    one_hot_seq.scatter_(1, integer_seq.unsqueeze(1), 1)
                    sequences.append(one_hot_seq)
                    labels.append(label_int)

                    # Save sequences and labels in batches
                    if len(sequences) >= batch_size:
                        self.save_batch(sequences, labels)
                        sequences = []
                        labels = []

            # Save any remaining sequences and labels
            if len(sequences) > 0:
                self.save_batch(sequences, labels)
                sequences = []
                labels = []

        print("finished caching")

    def save_batch(self, sequences, labels):
        # add dummy tensor to padd everything to uniform length
        sequences.append(torch.empty((30255, 4)))
        sequences = pad_sequence(sequences, batch_first=True)[:-1]
        with h5py.File(os.path.join(self.data_dir, "data.h5"), 'a') as f:
            if 'data' not in f:
                grp = f.create_group('data')
            else:
                grp = f['data']

            # Create a dataset for sequences if it doesn't exist
            if 'sequences' not in grp:
                seq_dataset = grp.create_dataset('sequences', data=sequences, maxshape=(None, None, None))
                seq_dataset.attrs.create('field_0_name', 'index')
                seq_dataset.attrs.create('field_1_name', 'sequence')
            else:
                seq_dataset = grp['sequences']
                seq_shape = seq_dataset.shape
                seq_dataset.resize((seq_shape[0] + len(sequences), seq_shape[1], seq_shape[2]))
                seq_dataset[seq_shape[0]:] = sequences

            # Extend the labels dataset
            if 'labels' not in grp:
                label_dataset = grp.create_dataset('labels', data=labels, maxshape=(None,))
                label_dataset.attrs.create('field_0_name', 'index')
                label_dataset.attrs.create('field_1_name', 'label')
            else:
                label_dataset = grp['labels']
                num_labels = label_dataset.shape[0]
                label_dataset.resize((num_labels + len(labels),))
                label_dataset[num_labels:] = labels

        gc.collect()

    def __getitem__(self, index):
        with h5py.File(os.path.join(self.data_dir, 'data.h5'), 'r') as f:
            seq_dataset = f['data']['sequences']
            label_dataset = f['data']['labels']
            seq, label = seq_dataset[index], label_dataset[index]
        a, b = seq.shape
        sqrt_ab = np.ceil(np.sqrt(a * b)).astype(int)
        square_matrix = np.zeros((sqrt_ab ** 2))
        square_matrix[:a * b] = seq.flatten()
        square_matrix = square_matrix.reshape((sqrt_ab, sqrt_ab))
        # convert to RGB
        square_matrix = np.repeat(square_matrix[:, :, np.newaxis], 3, axis=-1)

        if self.transform:
            square_matrix = self.transform(square_matrix)
        return square_matrix.to(torch.float), label

    def __len__(self):
        with h5py.File(os.path.join(self.data_dir, 'data.h5'), 'r') as f:
            seq_dataset = f['data']['sequences']
            return seq_dataset.shape[0]


def test_dataset():
    data_dir = r"/mnt/chromeos/MyFiles/university/8/Computer Architectures/data"
    dataset = CovidDataset(data_dir)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=4)
    with h5py.File(os.path.join(dataset.data_dir, 'data.h5'), 'r') as f:
        label_dataset = f['data']['labels'][:]
    arr = np.array(label_dataset)
    print(np.unique(arr, axis=0, return_counts=True))
