import torch
from torch.utils.data import Dataset
import proc_data as pd
import random
import math

SEED = 123456 # same as brevitas library

class time_series_dataset(Dataset):
    def __init__(self, data):
        # Initialize data, download, etc.
        self.data = data
        self.n_samples = len(data)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        tss = self.data[index]
        return tss.data, tss.class_label

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def gen_test_train_datasets(test_holdout):
    # initialize seed (keep consistent between runs)
    random.seed(SEED)
    torch.manual_seed(SEED)

    complete_input = pd.gen_input()
    in_len = len(complete_input)
    
    index_list = list(range(in_len))
    test_num = math.floor(test_holdout * in_len)
    test_inds = random.sample(index_list, test_num)
    train_inds = [x for x in index_list if x not in test_inds]

    test_data = []
    train_data = []
    for test_i in test_inds:
        test_data.append(complete_input[test_i])
    for train_i in train_inds:
        train_data.append(complete_input[train_i])

    train_dataset = time_series_dataset(train_data)
    test_dataset = time_series_dataset(test_data)

    return test_dataset, train_dataset
