import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math

# Building Modified_SPPLayer
class Modified_SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='average_pool'):
        super(Modified_SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type
        

    def forward(self, x):
        # num:number of samples; c:number of channels h:height w:width
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        c,num,h, w = x.size()   
#         print(x.size())
       for i in range(len(self.num_levels)):

        return x_flatten   
    
class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel,self).__init__() #PLAN1: input:9, output:45

        
        
    def forward(self,x):
        x=x.to(torch.float32)
        x=self.hidden1(x)
        x=self.active1(x)
        
        x=self.hidden2(x)
        x=self.active2(x)
        
        x=self.hidden3(x)
        x=self.active3(x)    
        
        x=self.hidden4(x)
        x=self.active4(x)
        output=self.regression(x)
        return output    
    
class SPPNet_s(nn.Module):
    def __init__(self):
        super(SPPNet_s, self).__init__()
        # self.output_num = [5,4,2,1]
        self.output_num = [4,2,1]
        
        # Define CNN layers for image processing

        
        # Define SPP layers

        
        # Define RNN layers for sequence processing


        # Initialize the weights using Xavier initialization

        
    def forward(self, imageData, sequenceData):
        # Process image data
 
        
        sequenceData=torch.squeeze(sequenceData)
        
        # Concatenate outputs from SPP and sequence
        concatenated = torch.cat((x, sequenceData), dim=0)        

        x = self.mlp(concatenated)
        
        return x

# Create an instance of the SPPNet
# net = SPPNet(num_features=45)

# Print the network architecture
# print(net)


