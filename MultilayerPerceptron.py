import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python import keras
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Dataset

data= pandas.read_csv("dataset/train.csv")
X= data.loc[:,'Gender':'Burn Rate']
X=X.dropna()
y= X['Burn Rate']
X= X.loc[:,'Gender':'Mental Fatigue Score']

column= X.columns.tolist()
for column_name in column[0:3]:
    X[column_name]=pandas.factorize(X[column_name],sort=True)[0]

X,X_test= train_test_split(X,test_size=0.3)
y,y_test= train_test_split(y,test_size=0.3)

print(X)
print(y)

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(6,3)
        self.fc2= nn.Linear(3,1)
        self.loss_func= F.mse_loss
    
    def forward(self,x):
        x= torch.relu(self.fc1(x.float()))
        return torch.sigmoid(self.fc2(x))

    def fit(self,data,lr,epochs):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            for xb,yb in data:
                out= self(xb)
                out= torch.squeeze(out,1)
                loss= self.loss_func(out,yb.float())
                loss.backward()
                # print(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                
X_torch= torch.tensor(X.values)
y_torch= torch.tensor(y.values)
print(X_torch)
print(y_torch)
train_ds= TensorDataset(X_torch,y_torch)

train_ld= DataLoader(train_ds,30)
print(train_ld)
model= Network()
model.fit(train_ld,0.05,100)

X_test= torch.tensor(X_test.values)
y_test= torch.tensor(y_test.values)
pred= model(X_test)
error= (pred-y_test)
accuracy= (error<=0.05).float().mean()
print(accuracy)
