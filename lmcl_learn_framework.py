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

# define NN models
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=400, kernel_size=(3, 5))
        self.fc1 = nn.Linear(2400, 20)
        self.fc2 = nn.Linear(20, 23)

    def forward(self, x):
        x = self.conv1(x)               # Convolution   -> 400 x 4 x 6
        x = F.max_pool2d(x, 2)          # MaxPool       -> 400 x 2 x 3
        x = F.relu(x)                   # ReLU          -> 400 x 2 x 3
        x = x.view(-1, 2400)            # Reshape       -> 1 x 2400
        x = self.fc1(x)                 # FC            -> 1 x 20 (Context invariant feature space)
        x = self.fc2(x)                 # FC            -> 1 x 23 (Context specific relation)
        x = F.log_softmax(x, dim=1)     # LogSoftMax    -> 1 x 23
        return x

batch_size = 50
num_epochs = 2000
learning_rate = 0.001
seq_len = 75

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def neural_network_train(data_loader):
    
    model = CNN().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
   
    model.train()
    for epoch in range(1, num_epochs + 1):
        for (data, target) in train_loader:
            # Convert to (index, time, channel, row, column).
            data = data[:,0,:,:,:].permute(0, 3, 1, 2).unsqueeze(2).float()
            data = data.to(device)
            data = data.view(-1, 1, 6, 10)

            target = torch.repeat_interleave(
                    target,
                    int(data.size()[0]) // int(target.size()[0]))
                    # repeat so that the same element should be in succession
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            # negative log likelihood loss https://pytorch.org/docs/stable/nn.html#nllloss
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
           
            print("Epoch = " + str(epoch) + ", Loss = " + str(loss.item()))

    return model

def neural_network_evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    for (data, target) in data_loader:
        # Convert to (index, time, channel, row, column).
        data = data[:,0,:,:,:].permute(0, 3, 1, 2).unsqueeze(2).float()
        data = data.to(device)
        data = data.view(-1, 1, 6, 10)

        target = target.to(device)

        output = model(data)
        output = output.view(-1, seq_len, 23)
        output = output.sum(dim = 1)
        output = output.max(dim = 1)[1]

        correct += output.eq(target).long().sum().item()
        total += output.size()[0]
    return float(correct) / total

if __name__ == "__main__":
    [train_ids, train_labels, test_ids, test_labels] = pickle.load(open('slide_6_10.pkl', 'rb'))
    training_dataset = Dataset(train_ids, train_labels)
    train_loader = data2.DataLoader(
            training_dataset, 
            batch_size=batch_size, 
            sampler=data2.RandomSampler(training_dataset),
            pin_memory=True)
    test_dataset = Dataset(test_ids, test_labels)
    test_loader = data2.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            sampler=data2.RandomSampler(test_dataset),
            pin_memory=True)

    #model = neural_network_train(train_loader)
    #torch.save(model, "lmcl.model")
    
    model = torch.load("lmcl.model").to(device)

    train_accuracy = neural_network_evaluate(model, train_loader)
    test_accuracy = neural_network_evaluate(model, test_loader)

    print("Accuracy (training) = " + str(train_accuracy))
    print("Accuracy (test)     = " + str(test_accuracy))
    print()
    
