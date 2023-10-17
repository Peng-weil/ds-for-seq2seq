import io
from logging import getLogger

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

logger = getLogger()


def get_order(s_len, o_type):
    s_order = []
    o_res = [i for i in range(s_len)]
    s_rank = int(len(o_res) ** 0.5)
    if not o_type == "SMR":
        s_matrix = np.array(o_res).reshape(s_rank, s_rank)

    if o_type == 'SMR':
        s_order.extend(o_res)
    elif o_type == 'SMC':
        o_res = s_matrix.T.reshape(1, -1).tolist()[0]
        s_order.extend(o_res)
    elif o_type == 'SMD':
        for diag_idx in range(s_rank):
            for m in range(diag_idx):
                s_order.append(s_matrix[m, diag_idx])
            for n in range(diag_idx, -1, -1):
                s_order.append(s_matrix[diag_idx, n])
    elif o_type == "counter-SMD":
        for diag_idx in range(s_rank):
            for m in range(diag_idx):
                s_order.append(s_matrix[diag_idx, m])
            for n in range(diag_idx, -1, -1):
                s_order.append(s_matrix[n, diag_idx])

    return s_order


class Matrix_Dataset(Dataset):
    def __init__(self, file_path, train, params):
        self.env_base_seed = params.env_base_seed

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size
        self.sort = params.sort_method

        # generation, or reloading from file
        logger.info(f"Loading data from {file_path} ...")
        with io.open(file_path, mode='r', encoding='utf-8') as f:
            if not train:
                lines = [line.rstrip() for line in f]
            else:
                lines = []
                for i, line in enumerate(f):
                    if i == params.reload_size:
                        break
                    lines.append(line.rstrip())

        self.data = [xy.split(',') for xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]

        logger.info(f"Loaded {len(self.data)} data from the disk.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x, y = self.data[idx]

        s_x_order = get_order(len(x), self.sort.split(",")[0])
        s_y_order = get_order(len(y), self.sort.split(",")[1])

        x_reorder = ''
        y_reorder = ''

        for idx in s_x_order:
            x_reorder += x[idx]
        for idx in s_y_order:
            y_reorder += y[idx]

        return x_reorder, y_reorder

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        words = ['<s>', '</s>', '<pad>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        id2word = {i: s for i, s in enumerate(words)}
        word2id = {s: i for i, s in id2word.items()}

        x, y = zip(*elements)
        x = [torch.LongTensor([word2id[w] for w in seq if w in word2id]) for seq in x]
        y = [torch.LongTensor([word2id[w] for w in seq if w in word2id]) for seq in y]
        x, x_len = self.batch_sequences(x)
        y, y_len = self.batch_sequences(y)

        return (x, x_len), (y, y_len)

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(1)
        assert lengths.min().item() > 2

        sent[0] = 0
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = 0

        return sent, lengths

# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description="test")
#     parser.add_argument('--num_workers', type=int, default=0)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--env_base_seed', type=int, default=0)
#     parser.add_argument('--reload_size', type=int, default=-1)
#
#     params = parser.parse_args()
#
#     data_path = '/public/home/pengwei2022/workspace/_matrix_deep/_diagonal_position_encoding/dataset/TRA/200_10_10_hy5cv'
#     matrix_dataset = Matrix_Dataset(
#         data_path,
#         True,
#         params)
#
#     train_size = int(0.8 * len(matrix_dataset))
#     test_size = int(0.2 * len(matrix_dataset))
#
#     trainDataset, testDataset = torch.utils.data.random_split(matrix_dataset, [train_size, test_size])
#
#     if params.reload_size != -1 and params.reload_size < len(trainDataset):
#         trainDataset = torch.utils.data.Subset(trainDataset,
#                                                np.random.choice(len(trainDataset), params.reload_size, replace=False))
#
#     trainLoader = DataLoader(matrix_dataset,
#                              timeout=(0 if params.num_workers == 0 else 1800),
#                              batch_size=params.batch_size,
#                              num_workers=(
#                                  params.num_workers if data_path is None or params.num_workers == 0 else 1),
#                              shuffle=True,
#                              collate_fn=matrix_dataset.collate_fn)
#
#     iterator = enumerate(trainLoader)
#     for _, data in iterator:
#         print(data)
