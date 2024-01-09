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
        c,num,h, w = x.size()   #[1,16,28,28]  ===>[1,336]
#         print(x.size())
        for i in range(len(self.num_levels)):
            level = self.num_levels[i]

            '''
            The equation is explained in the paper
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
        super(MLPmodel,self).__init__() #PLAN1: input:9, output:45
        #4*21+8  92
        self.hidden1=nn.Linear(13,256,bias=True)
        self.active1=nn.ReLU()
        
        self.hidden2=nn.Linear(256,176)
        self.active2=nn.ReLU()
        
        self.hidden3=nn.Linear(176,13)
        self.active3=nn.ReLU()
        
        self.hidden4=nn.Linear(13,128)
        self.active4=nn.ReLU()
        
        self.regression=nn.Linear(128,45)
        
        
    def forward(self,x):
        x=x.to(torch.float32)
        x_original=x
        x=self.hidden1(x)
        x=self.active1(x)
        
        x=self.hidden2(x)
        x=self.active2(x)
        
        x=self.hidden3(x)
        x=self.active3(x)        
        x=x+x_original
        
        x=self.hidden4(x)
        x=self.active4(x)
        output=self.regression(x)
        return output    
    
class SPPNet_s(nn.Module):
    def __init__(self):
        super(SPPNet_s, self).__init__()
        # self.output_num = [5,4,2,1]
        self.output_num = [2,1]
        
        # Define CNN layers for image processing
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.BN1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)
        
        # Define SPP layers
        self.spp = Modified_SPPLayer(self.output_num)
        
        # Define RNN layers for sequence processing
        # self.lstm = nn.LSTM(input_size=5, hidden_size=64, num_layers=1, batch_first=True)
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
        '''
        #to simplize the network, try not to use these CNN layers.
        x = self.pool1(F.relu(self.conv1(imageData))) #[1,32,21,21]        
        x = self.pool2(F.relu(self.conv2(x)))  #[1,16,9,9]
        x = self.conv3(x)  #[1,8,7,7]     
        x = F.relu(self.conv1(imageData))  #[1,1,32,32]=conv1=>[1,16,32,32]=relu=>[1,16,32,32]
        x = F.relu(self.BN1(self.conv2(x))) # [1,16,32,32]=conv2=>[1,64,30,30]=>BN1=>[1,64,30,30]=relu=>[1,64,30,30]
        x = self.conv3(x)  #([1, 16, 28, 28])
        '''
        x = self.spp(imageData) #16+4+1=21  21+25=46
        x = torch.squeeze(x)     
        
        sequenceData=torch.squeeze(sequenceData)
        
        # Concatenate outputs from CNN and sequence
        concatenated = torch.cat((x, sequenceData), dim=0)        

        x = self.mlp(concatenated)
        
        return x

# Create an instance of the SPPNet
# net = SPPNet(num_features=45)

# Print the network architecture
# print(net)


