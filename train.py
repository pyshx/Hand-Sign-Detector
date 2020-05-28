#!/usr/bin/env python
# coding: utf-8

# In[12]:


from torchsummary import summary
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Network as Net

import matplotlib.pyplot as plt

PATH = "trained_model"
# In[13]:


data_raw = pd.read_csv('train.csv', sep=",")
labels = data_raw['label']
data_raw.drop('label', axis=1, inplace=True)

valid_data_raw = pd.read_csv('validation.csv', sep=",")
valid_labels = valid_data_raw['label']
valid_data_raw.drop('label', axis=1, inplace=True)


# In[15]:


data = data_raw.values
labels = labels.values

valid_data = valid_data_raw.values
valid_labels = valid_labels.values

# In[16]:


pixels = data[10].reshape(50, 50)
plt.subplot(221)
sns.heatmap(data=pixels)

pixels = data[12].reshape(50, 50)
plt.subplot(222)
sns.heatmap(data=pixels)

pixels = data[20].reshape(50, 50)
plt.subplot(223)
sns.heatmap(data=pixels)

pixels = data[32].reshape(50, 50)
plt.subplot(224)
sns.heatmap(data=pixels)


# In[17]:


reshaped = []
for i in data:
    reshaped.append(i.reshape(1, 50, 50))
data = np.array(reshaped)

# In[18]:


x = torch.FloatTensor(data)
# print(x.shape)
y = torch.LongTensor(labels.tolist())


reshaped = []
for i in valid_data:
    reshaped.append(i.reshape(1, 50, 50))
valid_data_formatted = np.array(reshaped)

# In[18]:


valid_x = torch.FloatTensor(valid_data_formatted)
# print(x.shape)
valid_y = torch.LongTensor(valid_labels.tolist())

# In[20]:


device = torch.device("cpu")


#

# In[21]:


model = Net.Network().to(device)
summary(model, (1, 50, 50))


# In[22]:


net = Net.Network()

# optimizer = optim.SGD(net.parameters(), 1e-3, momentum=0.7)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()


# In[23]:


loss_log = []
acc_log = []
loss_log_valid = []

for e in range(20):
    for i in range(0, x.shape[0], 32):
        x_mini = x[i:i + 32]
        y_mini = y[i:i + 32]

        optimizer.zero_grad()
        net_out = net(Variable(x_mini))

        loss = loss_func(net_out, Variable(y_mini))
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            # pred = net(Variable(valid_data))
            loss_log.append(loss.item())

            net_valid = net(Variable(valid_x))
            loss_valid = loss_func(net_valid, Variable(valid_y))
            loss_log_valid.append(loss_valid.item())

            acc_log.append(net.evaluate(
                torch.max(net(Variable(valid_x[:100])).data, 1)[1], valid_y))

    print('Epoch: {} - Loss: {:.6f}'.format(e + 1, loss.item()))


# In[24]:


plt.figure(figsize=(10, 8))
plt.plot(loss_log[2:])
plt.plot(loss_log_valid)
plt.plot(acc_log)
plt.plot(np.ones(len(acc_log)), linestyle='dashed')
plt.show()
print(net.state_dict())
torch.save(net.state_dict(), PATH)
