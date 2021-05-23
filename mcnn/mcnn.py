
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import proc_data as pd
import gen_dataset
import utils

import time

pooling_factor = [2, 3, 5]
DATA_SIZE = 12
n_filters = 10
filter_sizes = [24, 16, 8]
filter_global = 24
num_classes = len(pd.LABELS.keys())

fc_vals = [30, 10]

SEED = 123456 # same as brevitas library

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
        self.conv1 = nn.Conv1d(DATA_SIZE, n_filters, filter_sizes[0])

        # smoothing
        self.conv_sm1 = nn.Conv1d(DATA_SIZE, n_filters, filter_sizes[1])
        self.conv_sm2 = nn.Conv1d(DATA_SIZE, n_filters, filter_sizes[2])
        
        # downsampling
        self.conv_dwn1 = nn.Conv1d(DATA_SIZE, n_filters, filter_sizes[1])
        self.conv_dwn2 = nn.Conv1d(DATA_SIZE, n_filters, filter_sizes[2])
        
        # self.pool1 = nn.MaxPool1d()
        self.pool1 = nn.AdaptiveMaxPool1d(n_filters)

        self.conv_global = nn.Conv1d(n_filters, n_filters, filter_global)
        self.pool_global = nn.AdaptiveMaxPool1d(n_filters)

        self.fc1 = nn.Linear(n_filters*filter_global, fc_vals[0])
        self.fc2 = nn.Linear(fc_vals[0], fc_vals[1])
        self.fc3 = nn.Linear(fc_vals[1], num_classes)

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

        print(x.shape)
        x = torch.transpose(x, 1, 2)

        print(x.shape)

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

        print(x.shape)

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

# class MCNNTrainer():
#     def __init__(self):

batch_size = 10
num_epochs = 4
learning_rate = 0.1

print("Getting Data...")

classes = pd.NUM_LABELS.keys()
test_dataset, train_dataset = gen_dataset.gen_test_train_datasets(0.1)
#train_dataset = torch.from_numpy(train_dataset).float()
#test_dataset = torch.from_numpy(test_dataset).float()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data setup complete...")

def train_model(model):
    random.seed(SEED)
    torch.manual_seed(SEED)

    # loss function + handles the softmax value (present in the final fully connected layer)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("Start Training...")
    n_total_steps = len(train_loader)
    print(n_total_steps)
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            # data = data.to(device)
            # labels = labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 5 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print('Finished Training')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    MOD_PATH = f'./cnn{num_epochs}_{timestr}.pth'
    did_save = utils.save_model(model, MOD_PATH)
    print("path: \t" + str(MOD_PATH))
    print("Saved Model: \t" + str(did_save))
    return model

def test(model):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(classes))]
        n_class_samples = [0 for i in range(len(classes))]
        for data, labels in test_loader:
            # data = data.to(device)
            # labels = labels.to(device)
            outputs = model(data)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
            # print(labels)
            for i in range(len(labels)):
                # print(i)
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(len(classes)):
            if n_class_samples[i] != 0:
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classes[i]}: {acc} %')

print("Creating Model...")
model = mcnn()
model.double()
print("Model created")

train_model(model)
test(model)
