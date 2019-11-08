import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils import data as data2
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import os


class Dataset(data2.Dataset):
    def __init__(self, list_IDs, labels):
        # initialize
        self.labels = labels
        self.list_IDs = list_IDs
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('slide_6_10/' + ID + '.pt')
        X.unsqueeze_(0)
        y = self.labels[ID]

        return X, y
    
    def get_X(self):
        X = []
        for i in range(len(self.list_IDs)):
            ID = self.list_IDs[i]
            # x = torch.load('slide_6_10/' + ID + '.pt').unsqueeze_(0)
            x = torch.load('slide_6_10/' + ID + '.pt')
            # convert to np array
            X.append(x.numpy())
        return np.array(X)
    
    def get_y(self):
        y = []
        for i in range(len(self.list_IDs)):
            ID = self.list_IDs[i]
            y.append(self.labels[ID])
        return np.array(y)


# define NN models
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,5))
        self.conv1_drop = nn.Dropout2d(p=0.8)

    def forward(self, x):
        #print('Conv:', x.size())
        x = self.conv1(x)
        # print('Conv', x.size())
        x = F.relu(F.max_pool2d(x, 2))
        # print('Pool', x.size())
        x = x.view(-1, 3*2*3)
        return x
    

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=3*2*3, 
            hidden_size=50, 
            num_layers=2,
            batch_first=True,
           dropout=0.8)
        
        self.linear = nn.Linear(50,23)
        self.hidden = []
        
        
    def init_hidden(self, h, c):
        self.hidden = (h, c)
        # Set initial hidden and cell states: initialize outside 
        #return (h, c) # (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))

    def forward(self, x):
        
        #print(x.size())
        batch_size, timesteps, C, H, W, sequence_size = x.size()
        #print(batch_size*timesteps,C, H, W, sequence_size)
        c_in = x.view(batch_size * timesteps*sequence_size, C, H, W)
        #print(c_in.size())
        
        c_out = self.cnn(c_in)
        #print(c_out.size())
        
        r_in = c_out.view(batch_size,sequence_size,-1)
        r_out, (h_n, h_c) = self.lstm( r_in, self.hidden)#(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
        r_out2 = self.linear(r_out[:, -1, :])

        return F.log_softmax(r_out2, dim=1)


class Network(nn.Module):
    '''Build a network that takes num_class as a param'''
    # max_num_class = 23 # in according to the dataset
    def __init__(self, num_class=23, num_new_class=0, hidden_size=50, lstm_layer=2, dropout=0.8):
        super(Network, self).__init__()
        self.num_class = num_class
        self.output_size = num_class + num_new_class
        if num_new_class==0:
            self.distillation = False
        else:
            self.distillation = True
        self.cnn = CNN()
        self.lstm = nn.LSTM(
            input_size=3*2*3,  # a specific way for data processing
            hidden_size=hidden_size, 
            num_layers=lstm_layer,
            batch_first=True,
           dropout=dropout)
        # print("linear layer ", hidden_size, num_class)
        self.linear = nn.Linear(hidden_size, self.output_size)
        self.hidden = []

        self.subnet = None

        if self.distillation:
            # TODO: create the subnet
            self.create_distillation_subnet(num_class=self.num_class, num_new_calss=0, hidden_size=hidden_size, lstm_layer=lstm_layer, dropout=dropout)
            # TODO: split the logits here
            pass
        
        
    def init_hidden(self, h, c):
        self.hidden = (h, c)
        # Set initial hidden and cell states: initialize outside 
        #return (h, c) # (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))

    def forward(self, x):
        
        #print(x.size()) 
        batch_size, timesteps, C, H, W, sequence_size = x.size()  # torch.Size([23, 1, 1, 6, 10, 75])
        #print(batch_size*timesteps,C, H, W, sequence_size)
        c_in = x.view(batch_size * timesteps*sequence_size, C, H, W)
        # print(c_in.size())
        
        c_out = self.cnn(c_in)
        # print(c_out.size())
        
        r_in = c_out.view(batch_size,sequence_size,-1)
        # print("r_in ", r_in.size())
        r_out, (h_n, h_c) = self.lstm(r_in, self.hidden) #(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
        # print("r_out", r_out[:, -1, :].size())
        r_out2 = self.linear(r_out[:, -1, :])
        # print("r_out2", r_out2)
        return F.log_softmax(r_out2, dim=1)

    def create_distillation_subnet(self, num_class=0, num_new_calss=0, hidden_size=50, lstm_layer=2, dropout=0.8):
        # create a network with computation for training
        
        with torch.no_grad():
            self.subnet = Network(num_class=num_class, num_new_class=0, hidden_size=hidden_size, lstm_layer=lstm_layer, dropout=dropout)
        # use tensor.detach() to stop back propagation
