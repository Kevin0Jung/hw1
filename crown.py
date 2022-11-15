"""
   crown.py
   COMP9444, CSE, UNSW
"""
#def graph_hidden(net, layer, node):
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, hid), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(hid, hid), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(hid, 1))
        
    def forward(self, input):
        x = self.layer1(input)
        self.hid1 = x
        x = self.layer2(x)
        self.hid2 = x
        x = self.layer3(x)
        x = torch.sigmoid(x)
        return x

class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, hid), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(hid, hid), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(hid, hid), nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(hid, 1))

    def forward(self, input):
        x = self.layer1(input)
        self.hid1 = x
        x = self.layer2(x)
        self.hid2 = x
        x = self.layer3(x)
        self.hid3 = x
        x = self.layer4(x)
        x = torch.sigmoid(x)
        return x
        

#implements a 3-layer densely connected neural network.
class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, num_hid), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(num_hid+2, num_hid), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(num_hid+num_hid+2, 1))


    def forward(self, input):
        x = self.layer1(input)
        self.hid1 = x
        x = torch.cat((input, x), 1)
        x = self.layer2(x)
        self.hid2 = x
        x = torch.cat((x, self.hid1,input), 1)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        return x
       
