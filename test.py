import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import Network as Net

PATH = "trained_model"

test_data_raw = pd.read_csv('test.csv', sep=",")
labels_test = test_data_raw['label']
test_data_raw.drop('label', axis=1, inplace=True)
test_data = test_data_raw.values
labels_test = labels_test.values
reshaped_test = []
for i in test_data:
    reshaped_test.append(i.reshape(1,50,50))
test_data = np.array(reshaped_test)
test_x = torch.FloatTensor(test_data)
test_y = torch.LongTensor(labels_test.tolist())


net = Net.Network()
net.load_state_dict(torch.load(PATH))
net.eval()

predictions = net(Variable(test_x))
net.test(torch.max(predictions.data, 1)[1], test_y)