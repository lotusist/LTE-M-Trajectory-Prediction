# -*- coding: utf-8 -*-

from io import open
import os.path
from os import path
import random
import numpy as np

import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
BatchSize = 128

_PATH = './bi-lstm/'
NETWORK = 'lte_'
if NETWORK == 'ais_': filename = 'ais_grid_sample.csv'
else: filename = 'lte_concat_1M.csv'

class TrajectoryDataset(Dataset):
    
    def __init__(self, csv_file= _PATH + filename):

        self.csv_file = csv_file
        self.X_frames = []
        self.Y_frames = []
        self.load_data()
        self.normalize_data()
    
    def __len__(self):
        return len(self.X_frames)
    
    def __getitem__(self, idx):
        single_data = self.X_frames[idx]
        single_label = self.Y_frames[idx]
        return (single_data, single_label)
    
    def load_data(self):
        dataS = pd.read_csv(self.csv_file)
        max_shipnum = len(dataS.szSrcID.unique())
        for _, id in enumerate(dataS.szSrcID.unique()):
            print('{0} and {1}'.format(_, max_shipnum))
            frame_ori = dataS[dataS.szSrcID == id]
            frame = frame_ori[['dLat', 'dLon', 'dSOG', 'dCOG']]
            total_frame_data = np.asarray(frame)
            if(total_frame_data.shape[0]<200): # 20 vessels in lte (out of 977) are filtered out. 
                continue
            
            X = total_frame_data[:-29,:]
            Y = total_frame_data[29:,:4]
            
            
            count = 0
            for i in range(X.shape[0]-100):
                if random.random()>0.2:
                    continue
                j = i-1
                if count>20:
                    break
                self.X_frames = self.X_frames + [X[i:i+100,:]]
                self.Y_frames = self.Y_frames + [Y[i:i+100,:]]
                count = count+1
    def normalize_data(self):
        A = [list(x) for x in zip(*(self.X_frames))]
        A = torch.tensor(A)
        A = A.view(-1,A.shape[2])
        print('A:',A.shape)
        self.mn = torch.mean(A,dim=0)
        self.range = (torch.max(A,dim=0).values-torch.min(A,dim=0).values)/2.0
        self.range = torch.ones(self.range.shape,dtype = torch.double)
        self.std = torch.std(A,dim=0)
        # self.X_frames = [torch.tensor(item) for item in self.X_frames]
        # self.Y_frames = [torch.tensor(item) for item in self.Y_frames]
        self.X_frames = [(torch.tensor(item)-self.mn)/(self.std*self.range) for item in self.X_frames]
        self.Y_frames = [(torch.tensor(item)-self.mn[:4])/(self.std[:4]*self.range[:4]) for item in self.Y_frames]

def get_dataloader():
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''
    if path.exists(_PATH + NETWORK + "my_dataset.pickle"):
        with open(_PATH + NETWORK + 'my_dataset.pickle', 'rb') as data:
            dataset = pickle.load(data)
        print("dataset pickle already exists! :", _PATH + NETWORK + "my_dataset.pickle")
    else:
        dataset = TrajectoryDataset()
        with open(_PATH + NETWORK + 'my_dataset.pickle', 'wb') as output:
            pickle.dump(dataset, output)
        print("dataset has been made and saved! :", _PATH + NETWORK + 'my_dataset.pickle')
        

    #split dataset into train test and validation 7:2:1
    num_train = (int)(dataset.__len__()*0.7)
    num_test = (int)(dataset.__len__()*0.9) - num_train
    num_validation = (int)(dataset.__len__()-num_test-num_train)
    train, test, validation = torch.utils.data.random_split(dataset, [num_train, num_test, num_validation])
    train_loader = DataLoader(train, batch_size=BatchSize, shuffle=True)
    test_loader = DataLoader(test, batch_size=BatchSize, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=BatchSize, shuffle=True)
    return (train_loader, test_loader, validation_loader, dataset)
