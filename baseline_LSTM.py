# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import time
import math
import random
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as dist
from dataPrepare import *

torch.manual_seed(0)
BatchSize = 128

_PATH = './bi-lstm/'
NETWORK = '_lte_'
TRAN_TAG = False # change True to train the model. change False to test the model. 
EXPORT_TAG = True # change False if you want to skip saving results in csv files 


MAX_LENGTH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('ok')
Training_generator, Test, Valid, WholeSet= get_dataloader()

class NNPred(nn.Module):
    def __init__(self, input_size, output_size,hidden_size,batch_size, dropout=0.05):
        super(NNPred, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.in2lstm = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,num_layers=self.num_layers,bidirectional=False,batch_first=True,dropout =0.1)
        self.in2bilstm = nn.Linear(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size//2,num_layers=self.num_layers,bidirectional=True,batch_first=True,dropout =0.1)
    
        self.fc0 = nn.Linear(256,128)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64 ,output_size)
        self.in2out = nn.Linear(input_size, 64)
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()

        
    def forward(self, input):
        #input = tensor shape[batchsize, len, num_features]
 
        bilstm_out,_= self.lstm(self.in2bilstm(input))
        
        lstm_out,_= self.lstm(self.in2lstm(input))
        out = F.tanh(self.fc0(lstm_out+bilstm_out))
        out = F.tanh(self.fc1(out))
        out =  out + self.in2out(input)
        output = self.fc2(out)# range [0 -> 1 ]
        return output

def trainIters(encoder, n_iters, print_every=1000, plot_every=1, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')
    pltcount = 0
    prtcount = 0
    cp = 0
    for iter in range(1, n_iters + 1):
        if iter%50==1:
            cp = cp+1
            torch.save(encoder.state_dict(), _PATH + str(cp) + NETWORK + 'checkpoint.pth.tar')
        for local_batch, local_labels in Training_generator:
            if local_batch.shape[0]!=BatchSize:
                continue
            pltcount = pltcount+1
            prtcount = prtcount+1
            encoder.zero_grad()
            
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(device)
            
            predY = encoder(local_batch)
            loss = criterion(predY[:,-30:,:2],local_labels[:,-30:,:2]).to(device)
            loss.backward()
            encoder_optimizer.step()
            
            ls =  loss.detach().item()
            print_loss_total += ls
            plot_loss_total += ls
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / prtcount
            print_loss_total = 0
            prtcount = 0
            print('%s (%d %d%%) %f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / pltcount
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            pltcount = 0
    return plot_losses

def Eval_net(encoder):
    count = 0
    pred_errors = []
    true_waypoints = []
    pred_waypoints = []
    
    def denormalize(output, mn, std, range_val):
        return (output * (std * range_val)) + mn

    for local_batch, local_labels in Test:
        if local_batch.shape[0] != BatchSize:
            continue
        count = count + 1
        local_batch = local_batch.to(device)
        local_labels = local_labels.to(device)
        predY = encoder(local_batch)
        ls = calcu_XY(predY, local_labels)
        # pred_errors.append(ls.item())  # Assuming ls is a scalar tensor
        pred_errors.append(ls)  # Assuming ls is a scalar tensor


        # Move tensors to CPU and denormalize
        local_batch = local_batch.detach().cpu()
        local_batch = denormalize(local_batch, WholeSet.mn[:4], WholeSet.std[:4], WholeSet.range[:4])
        true_waypoints.append(local_batch.numpy())

        local_labels = local_labels.detach().cpu()
        local_labels = denormalize(local_labels, WholeSet.mn[:4], WholeSet.std[:4], WholeSet.range[:4])
        true_waypoints.append(local_labels.numpy())  # Convert to numpy array

        predY = predY.detach().cpu()
        predY = denormalize(predY, WholeSet.mn[:4], WholeSet.std[:4], WholeSet.range[:4])
        pred_waypoints.append(predY.numpy())  # Convert to numpy array

    print("prediction error (Sum of MSE):", sum(pred_errors))
    print("batches in Test:", count)
    return pred_errors, true_waypoints, pred_waypoints

        
def calcu_XY(predY,labelY):
    #input: [batchsize len features]; features:[x,y,cog,sog]
    criterion = nn.MSELoss(reduction='sum')
    loss = criterion(predY[:, -30:, :2], labelY[:, -30:, :2])
    return loss.item()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=100)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(_PATH + NETWORK + 'showPlot.png')
    plt.show()


# main

train_iter = iter(Training_generator)
x, y = next(train_iter)
print(x.shape)
hidden_size = 256
Prednet = NNPred(x.shape[2], y.shape[2],hidden_size, BatchSize)

print(device)


if TRAN_TAG:
    if path.exists(_PATH + NETWORK + "checkpoint.pth.tar"):
        Prednet.load_state_dict(torch.load(_PATH + NETWORK + 'checkpoint.pth.tar'))
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)
    plot_losses = trainIters(Prednet, 10, print_every=2)
    torch.save(Prednet.state_dict(), _PATH + NETWORK + 'checkpoint.pth.tar')
    showPlot(plot_losses)
else:
    Prednet.load_state_dict(torch.load(_PATH + NETWORK + 'checkpoint.pth.tar'))
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)
    Prednet.eval()
    
    pred_errors, true_waypoints, pred_waypoints = Eval_net(Prednet)

    if EXPORT_TAG:
        true_waypoints = [pd.DataFrame(tensor.tolist()) for tensor in true_waypoints]
        pred_waypoints = [pd.DataFrame(tensor.tolist()) for tensor in pred_waypoints]
        for i, waypoints in enumerate(true_waypoints):
            waypoints.to_csv(f'./results/true_waypoints_{i}.csv', index=False)
        for i, waypoints in enumerate(pred_waypoints):
            waypoints.to_csv(f'./results/pred_waypoints_{i}.csv', index=False)
