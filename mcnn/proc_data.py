import os
import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader

LOGGER_ON = False

LABELS = {
    "dws": "downstairs",
    "ups": "upstairs",
    "sit": "sitting",
    "std": "standing",
    "wlk": "walking",
    "jog": "jogging"
}

NUM_LABELS = {
    "dws": 0,
    "ups": 1,
    "sit": 2,
    "std": 3,
    "wlk": 4,
    "jog": 5
}

whole_path = str(os.getcwd()) + "/"
msd_path = whole_path + "motionsense/data/"

A_MD = "A_DeviceMotion_data/A_DeviceMotion_data/"

FILE_LABELS = {
    "dws": [1, 2, 11],
    "ups": [3, 4, 12],
    "sit": [5, 13],
    "std": [6, 14],
    "wlk": [7, 8, 15],
    "jog": [9, 16]
}

class time_series_sample:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.class_label = label[0]

# class WineDataset(Dataset):

#     def __init__(self):
#         # Initialize data, download, etc.
#         # read with numpy or pandas
#         xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
#         self.n_samples = xy.shape[0]

#         # here the first column is the class label, the rest are the features
#         self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
#         self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

#     # support indexing such that dataset[i] can be used to get i-th sample
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     # we can call len(dataset) to return the size
#     def __len__(self):
#         return self.n_samples

def gen_input():
    input_data = []
    if LOGGER_ON:
        c = 0
    for k in FILE_LABELS.keys():
        for num_id in FILE_LABELS[k]:
            file_name = str(msd_path + A_MD + str(k) + "_" + str(num_id) + "/")

            for i in list(range(1,25)):
                try:
                    a_data = np.genfromtxt(file_name + "sub_" + str(i) + ".csv", delimiter=',', skip_header=1)
                    label = tuple([k, num_id, i])
                    new_tss = time_series_sample(a_data, label)
                    input_data.append(new_tss)
                    if LOGGER_ON:
                        c = c + 1
                        print(str(label) + "    " + str(c))
                except:
                    pass
    return input_data
