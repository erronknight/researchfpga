import os
import numpy as np

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
                    a_data = a_data[0:4000,1:]
                    if len(a_data) < 4000:
                        df = (4000 - len(a_data))
                        a_data.extend([[0]* len(a_data[0])] * df)
                    new_tss = time_series_sample(a_data, label)
                    input_data.append(new_tss)
                    if LOGGER_ON:
                        c = c + 1
                        print(str(label) + "    " + str(c))
                except:
                    pass
    return input_data

def data_stats(data):
    # lengths
    ave = 0
    maxi = 0
    mini = 0
    for datum in data:
        ave = ave + len(datum.data)
        if maxi < len(datum.data):
            maxi = len(datum.data)
        if mini == 0:
            mini = len(datum.data)
        if mini > len(datum.data):
            mini = len(datum.data)
    ave = ave / len(data)

    print("average: \t" + str(ave))
    print("max val: \t" + str(maxi))
    print("min val: \t" + str(mini))

def data_dist(data):
    # 300, ... , 17000
    # 334, 50
    aaa = []
    distro = [0] * 50
    for datum in data:
        le = len(datum.data)
        i = (le - 300)//334
        distro[i] = distro[i] + 1
        aaa.append(le)
    print(distro)
    print(np.median(aaa))
    print(sorted(aaa))

# Identity (1)
# Smoothing (moving average) (2)
# Downsampling (2)

def smooth_data(data, data_size, wdw):
    sm_data = []
    for d in data:
        sm_d = []
        window = [[]] * data_size
        # # beginning
        # dslice = d.data[0:(wdw//2)]
        # dslice = np.transpose(dslice)
        # for x in data_size:
        #     window[x] = dslice[x]

        for row in d.data:
            sm_row = [None] * data_size
            for i,val in enumerate(row):
                # if i > (wdw//2):
                window[i].append(val)
                if len(window[i]) > wdw:
                    window[i].pop(0)
                sm_row[i] = [np.mean(window[i])]
            sm_d.append(sm_row)
        
        sm_tss = time_series_sample(sm_d, d.label)
        sm_data.append(sm_tss)
    return sm_data

def downsample_data(data, k_value):
    # every kth value
    dwns_data = []
    for d in data:
        dwns_d = []
        count = 0
        d_len = len(d.data)
        while count < d_len:
            row = d.data[count]
            dwns_d.append(row)
            count = count + k_value
        dwns_tss = time_series_sample(dwns_d, d.label)
        dwns_data.append(dwns_tss)
    return dwns_data

def smooth_data_tss(d, data_size, wdw):
    sm_d = []
    window = [[]] * data_size

    for row in d.data:
        sm_row = [None] * data_size
        for i,val in enumerate(row):
            # if i > (wdw//2):
            window[i].append(val)
            if len(window[i]) > wdw:
                window[i].pop(0)
            sm_row[i] = [np.mean(window[i])]
        sm_d.append(sm_row)
    
    sm_tss = time_series_sample(sm_d, d.label)
    return sm_tss

def downsample_data_tss(d, k_value):
    # every kth value
    dwns_d = []
    count = 0
    d_len = len(d.data)
    while count < d_len:
        row = d.data[count]
        dwns_d.append(row)
        count = count + k_value
    dwns_tss = time_series_sample(dwns_d, d.label)
    return dwns_tss

# average:        3924.625
# max val:        16424
# min val:        377

'''
dat = gen_input()
data_stats(dat)
data_dist(dat)

smdat = smooth_data(dat, 12, 5)
dwndat = downsample_data(dat, 5)

data_stats(smdat)
print("---")
data_stats(dwndat)
'''

# print(len(dat[0].data[0]))
# data_stats(dat)