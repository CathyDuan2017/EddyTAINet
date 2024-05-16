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
            '''
            The equation is explained:
            Based on the relationship between the visual field parameters of the input and output layers in the neural network,
the output size is determined by the input feature size, convolution kernel, step size and padding together. The SPP
network module, on the other hand, determines the convolution kernel, step size and padding parameters in turn from
the fixed pooled size. Thus the setup process is shown in Fig1_SPPalg.jpg
            '''
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))
            
            # update input data with padding
            zero_pad = torch.nn.ZeroPad2d((pooling[1],pooling[1],pooling[0],pooling[0]))
            x_new = zero_pad(x)
            
            # update kernel and stride
            h_new = 2*pooling[0] + h
            w_new = 2*pooling[1] + w
            
            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))            
            
            # choose the pool type 
            if self.pool_type == 'max_pool':
                try:
                    tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
                except Exception as e:
                    print(str(e))
                    print(x.size())
                    print(level)
            else:
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)  
              
            # expand and joint
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten   
    
class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel,self).__init__() # input:8+16+4+1  or 8+4+1, output:45
        self.hidden1=nn.Linear(13,256,bias=True)
        self.active1=nn.ReLU()
        
        self.hidden2=nn.Linear(256,128)
        self.active2=nn.ReLU()
        
        self.hidden3=nn.Linear(128,64)
        self.active3=nn.ReLU()
        
        self.hidden4=nn.Linear(64,128)
        self.active4=nn.ReLU()  
        
        self.regression=nn.Linear(128,45)
        
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
        
        x=self.hidden5(x)
        x=self.active5(x)
        output=self.regression(x)
        return output    
    
class SPPNet_s(nn.Module):
    def __init__(self):
        super(SPPNet_s, self).__init__()
        # self.output_num = [2,1] #or use two space pools
        self.output_num = [4,2,1]
        
        # Define CNN layers for image processing - Optional steps     
        # Define SPP layers
        self.spp = Modified_SPPLayer(self.output_num)
        # Define RNN layers for sequence processing -Optional steps
        # Define the MLP layers 
        self.mlp = MLPmodel() 
        # Initialize the weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
    def forward(self, imageData, sequenceData):
        # Process image data
        # CNN processing 
        # RNN processing
        # MLP processing 
        x = self.spp(imageData) 
        x = torch.squeeze(x)
        sequenceData=torch.squeeze(sequenceData)        
        # Concatenate outputs from SPP and sequence
        concatenated = torch.cat((x, sequenceData), dim=0)  
        x = self.mlp(concatenated)        
        return x
