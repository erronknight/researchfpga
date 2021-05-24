
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import proc_data as pd
import gen_dataset

import time

pooling_factor = [2, 3, 5]
DATA_SIZE = 12
n_filters = 10
filter_sizes = [16, 8, 4]
filter_global = 16
glob_filt = 20
num_classes = len(pd.LABELS.keys())

fc_vals = [30, 10]

SEED = 123456 # same as brevitas library

# load in model
def load_model(model_path):
    model = mcnn()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# save model
def save_model(model, model_path):
    try:
        torch.save(model.state_dict(), model_path)
        return True
    except:
        return False

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

        self.conv_global = nn.Conv1d(n_filters, glob_filt, filter_global)
        self.pool_global = nn.AdaptiveMaxPool1d(n_filters)

        self.fc1 = nn.Linear(200, fc_vals[0])
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

        #print(x.shape)
        #print(x)
        x_sm1 = pd.smooth_data_ten(x, DATA_SIZE, self.window1)
        x_sm2 = pd.smooth_data_ten(x, DATA_SIZE, self.window2)
        x_dwn1 = pd.downsample_data_ten(x, self.k1)
        x_dwn2 = pd.downsample_data_ten(x, self.k2)

        print("completed branching")

        x = torch.transpose(x, 1, 2)
        x_sm1 = torch.transpose(x_sm1, 1, 2)
        #x_sm2 = torch.transpose(x_sm2, 1, 2)
        #print(x_dwn1)
        #print(torch.FloatTensor(x_dwn1).shape)
        x_dwn1 = torch.transpose(x_dwn1, 1, 2)
        #x_dwn2 = torch.transpose(x_dwn2, 1, 2)


        #print(x.shape)

        # x identity (1)
        x1 = self.pool1(self.activation(self.conv1(x)))

        #print(x_sm1.shape)
        # x smoothing (moving average) (2)
        x2 = self.conv_sm1(torch.squeeze(x_sm1))
        #print(x2.shape)
        x2 = self.activation(x2)
        #print(x2.shape)
        x2 = self.pool1(x2)
        #print(x2.shape)
        #x3 = torch.squeeze(x_sm2)
        #x2 = self.pool1(self.activation(self.conv_sm1(x_sm1)))
        #x3 = self.pool1(self.activation(self.conv_sm2(x3)))

        # x downsampling (every kth item) (2)
        x4 = self.pool1(self.activation(self.conv_dwn1(x_dwn1)))
        #x5 = self.pool1(self.activation(self.conv_dwn2(x_dwn2)))

        # conv1d and maxpool for each

        # concatenate
        #xcat = torch.cat((x1, x2, x3, x4, x5), 2)
        xcat = torch.cat((x1,x2,x4), 2)

        #print(xcat.shape)

        # conv1d and maxpool
        x = self.pool_global(self.activation(self.conv_global(xcat)))

        #print(x.shape)

        # TODO
        x = x.view(10, 200)

        # 2 fc then fc with softmax (self.fullyconnectedlayer(x))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x


# generate model (base)

# train model

# test model

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
                print (f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")
                #print(str(epoch) + "  " + str(i) + "  " + str(loss.item()))
        if epoch % 5 == 0:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            MOD_PATH = f'./cnn{num_epochs}_{epoch+1}_{timestr}.pth'
            did_save = save_model(model, MOD_PATH)
            print("path: \t" + str(MOD_PATH))
            print("Saved Model: \t" + str(did_save))

        
    print('Finished Training')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    MOD_PATH = f'./cnn{num_epochs}_{timestr}.pth'
    did_save = save_model(model, MOD_PATH)
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

batch_size = 10
num_epochs = 50 
learning_rate = 0.02

if __name__ == 'main':

    print("Getting Data...")

    classes = tuple(pd.NUM_LABELS.keys())
    test_dataset, train_dataset = gen_dataset.gen_test_train_datasets(0.1)
    #train_dataset = torch.from_numpy(train_dataset).float()
    #test_dataset = torch.from_numpy(test_dataset).float()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data setup complete...")

    print("Creating Model...")
    model = mcnn()
    model.double()

    model = load_model("./cnn50_46_20210524-104056.pth")

    print("Model created")

    # train_model(model)
    test(model)
