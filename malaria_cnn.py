#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 00:41:21 2021

@author: swain_asish
"""
#%% library import 
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

#%% find mean and std for 3000 random data
path='/home/minato/Desktop/itsok/DL/datasets/malaria/cell_images'
transform= transforms.ToTensor()
train_dt=datasets.ImageFolder(root= path+'/train',transform=transform)
means=torch.tensor([0,0,0],dtype=torch.float32)
stds=torch.tensor([0,0,0],dtype=torch.float32)  
nos=np.random.randint(0,20000,3000)
for i in nos:
    image=train_dt[i][0]                              #we don't need label here
    mean=torch.mean(image,axis=(1,2))
    std=torch.std(image,axis=(1,2))
    means+=mean
    stds+=std
means=means/3000
stds=stds/3000 
print(f"mean:{means}, std:{stds}")
#%% image transformation of train and test data & sumup the data
train_transformation=transforms.Compose([
                      transforms.RandomHorizontalFlip(p=0.5),
                      transforms.Resize(250),
                      transforms.CenterCrop(225),
                      transforms.ToTensor(),
                      transforms.Normalize(means,stds)   #put the mean and std
                      
   ])

test_transformation=transforms.Compose([
                      transforms.Resize(250),
                      transforms.CenterCrop(225),
                      transforms.ToTensor(),
                      transforms.Normalize(means,stds) 
   ])

train_dt=datasets.ImageFolder(root= path+'/train',transform=train_transformation)
test_dt=datasets.ImageFolder(root=path+'/test',transform=test_transformation)
print(train_dt.classes)
print(test_dt.classes)
print(train_dt.class_to_idx)
#%% load into dataloader 
train_loader= DataLoader(dataset=train_dt,shuffle=True,batch_size=10)
test_loader=DataLoader(dataset=test_dt,shuffle=True,batch_size=100)
#%%  Build the CNN
class malaria_CNN(nn.Module):
   def __init__(self):
      super(malaria_CNN,self).__init__()
      
      #Convolution layers 
      conv_kernels=[16,32,64,128,64]
      conv_layers=[]
      input=3
      for n,i in enumerate(conv_kernels):
         conv_layers.append(nn.Conv2d(in_channels=input,out_channels=i,
                                      kernel_size=(5,5),stride=1,padding=1))
         conv_layers.append(nn.ReLU())
         conv_layers.append(nn.Dropout2d(p=0.3,inplace=True))
         if n%2==0:
            conv_layers.append(nn.MaxPool2d(3))
         input=i
      self.Conv_Seq=nn.Sequential(*conv_layers)      
      
      #Linear layers
      linear_neurons=[1500,600,40]
      linear_layers=[]
      input=2304
      for i in linear_neurons:
         linear_layers.append(nn.Linear(in_features=input,out_features=i))
         linear_layers.append(nn.ReLU(inplace=True))
         linear_layers.append(nn.Dropout2d(p=0.5))
         input=i
      linear_layers.append(nn.Linear(input,2))    
      self.Linear_Seq= nn.Sequential(*linear_layers)
                               
   def forward(self,X):
      X = self.Conv_Seq(X)
      samples=X.shape[0]
      X = X.view(samples,-1)
      X = self.Linear_Seq(X)
      return X
   
#%% Build Model , Loss function and optimiser(which will use for backprop)
model=malaria_CNN()
loss_fun=nn.CrossEntropyLoss()
learn_rate=0.001
optimiser=torch.optim.Adam(model.parameters(),lr=learn_rate)

#%% Total no of parametrs used by our model
total_params=0
for params in model.parameters():
    x=params.numel()
    total_params+=x
print(f'total parameters used by this model for 1 iteration: {total_params}') 


#%% GPU implement
import torch
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= model.to(device)
print(device)

#%% Prediction of train_loader,test_loader
import time
epochs = 6

losses = []
start=time.time()
for epoch in range(epochs):
  with torch.no_grad():
      accuracies=[]
      for j,(data,label)in enumerate(test_loader):
          data,label=data.to(device),label.to(device)
          ypred=model.forward(data)
          ypred=torch.max(ypred,1)[1]
          correct=sum(ypred==label)
          accu=correct/len(ypred)
          accuracies.append(accu)
          if j>40:
            break
      mean_acc= sum(accuracies)/len(accuracies)
      print(f'Epoch:{epoch} Accuracy: {mean_acc*100} percent ')
  for i,(data1,label1) in enumerate(train_loader):
      data1,label1 = data1.to(device),label1.to(device)
      ypred = model.forward(data1)
      loss = loss_fun(ypred, label1)
      losses.append(loss)
      if i%100==0:
        print(f'Epoch:{epoch}, SubEpoch :{i} loss: {loss}')
      model.zero_grad()
      loss.backward()
      optimiser.step()
end=time.time()
print(f'Toal time {(end-start)/60} minute')     
#%%
accura=[]
with torch.no_grad():
  for dt,lb in test_loader:
    dt,lb=dt.to(device),lb.to(device)
    yp=model.forward(dt)
    yp = torch.max(yp,1)[1]
    corre= sum(yp==lb)
    acc= corre/len(lb)
    accura.append(acc)
    
 
























