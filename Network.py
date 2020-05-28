import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(10, 20, 3)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(20, 30, 3)
        self.dropout1 = nn.Dropout2d()

        self.fc3 = nn.Linear(30*9*9, 40)
        self.fc4 = nn.Linear(40, 4)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # print(x.shape)
        x = x.view(-1, 30*9*9)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return self.softmax(x)

    def test(self, predictions, labels):

        self.eval()
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1

        acc = correct / len(predictions)
        print("Correct predictions: %5d / %5d (%5f)" %
              (correct, len(predictions), acc))

    def evaluate(self, predictions, labels):

        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1

        acc = correct / len(predictions)
        return(acc)
