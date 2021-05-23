
import torch
import torch.nn as nn
import torch.nn.functional as F

import proc_data as pd

pooling_factor = [2, 3, 5]
DATA_SIZE = 12
n_filters = 10


# Define model
class mcnn(nn.Module):
    def __init__(self):
        super(mcnn, self).__init__()

        self.k1 = 5
        self.k2 = 10
        self.window1 = 10
        self.window2 = 20
        self.activation = F.relu

        # TODO
        # identity
        self.conv1 = nn.Conv1d()

        # smoothing
        self.conv_sm1 = nn.Conv1d()
        self.conv_sm2 = nn.Conv1d()
        
        # downsampling
        self.conv_dwn1 = nn.Conv1d()
        self.conv_dwn2 = nn.Conv1d()
        
        self.pool1 = nn.MaxPool1d()

        self.conv_global = nn.Conv1d()
        self.pool_global = nn.MaxPool1d()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()

        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        # set up so that each (technically) has 3
        # mult-freq [identity, smoothed, smoothest]
        # mult-scale [identity, medium, small]

        # x identity (1)
        x1 = self.pool1(self.activation(self.conv1(x)))

        # x smoothing (moving average) (2)
        x_sm1 = pd.smooth_data_tss(x, DATA_SIZE, self.window1)
        x_sm2 = pd.smooth_data_tss(x, DATA_SIZE, self.window2)
        x2 = self.pool2(self.activation(self.conv_sm1(x_sm1)))
        x3 = self.pool2(self.activation(self.conv_sm2(x_sm2)))

        # x downsampling (every kth item) (2)
        x_dwn1 = pd.downsample_data_tss(x, self.k1)
        x_dwn2 = pd.downsample_data_tss(x, self.k2)
        x4 = self.pool2(self.activation(self.conv_dwn1(x_dwn1)))
        x5 = self.pool2(self.activation(self.conv_dwn2(x_dwn2)))

        # conv1d and maxpool for each

        # concatenate
        xcat = torch.cat((x1, x2, x3, x4, x5))

        # conv1d and maxpool
        x = self.pool_global(self.activation(self.conv_global(xcat)))

        # TODO
        x = x.view(-1, )

        # 2 fc then fc with softmax (self.fullyconnectedlayer(x))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x


# generate model (base)

# train model

# test model
