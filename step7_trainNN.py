import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from step6_MultiSource import SPPNet
from step6_MultiSource_simple import SPPNet_s
import numpy as np
from torch.autograd import Variable
import json
from torch.utils.data import random_split
from sklearn.metrics import r2_score  
import matplotlib.pyplot as plt    
import pickle
# a dataset class that handles data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


AE_filePath = r'xxAE.json'

indexArgo=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
          23,28,33,38,43,48,53,58,63,68,73,78,83,88,93,98,
          108,118,128,138,148,158,168,178,188,198]

with open(AE_filePath,'r') as fp:
    dateitems=json.load(fp)
    ae_firstFlag=1
    iterflat=0
    for date_ in dateitems: 
        oneDayValues=dateitems[date_]
        for item in oneDayValues:
            x1_year=item['x1_year']
            x2_diffday=item['x2_diffday']
            x3_argo_lon=item['x3_argo_lon']
            x4_argo_lat=item['x4_argo_lat']  
            x5_eddy_radius=item['x5_eddy_radius']
            x6_eddy_eke=item['x6_eddy_eke']
            x7_eddy_amp=item['x7_eddy_amp']
            x8_eddy_nd=item['x8_eddy_nd']
            x10_cost=item['xothers_cost']
            x11_sint=item['xothers_sint']
            x12_cost2t=item['xothers_cos2t']
            x13_sin2t=item['xothers_sin2t'] 
            
            x_sst_len=item['SST_lenth'] 
            x_sst_data=item['SST_restore'] 
            sstnp_array=np.array(x_sst_data) 

            # Find invalid cells (larger than 30): there would be invalid SSTA values, need to process them. 
            # After replacing the invalid values with neighbourhood values, there may still be invalid values in the boundary region, which can affect the training process and can be removed as appropriate
            invalid_indices = np.where(sstnp_array > 30)
            [imgLen,imgWid]=sstnp_array.shape
            for i in range(len(invalid_indices[0])):
                x, y = invalid_indices[0][i], invalid_indices[1][i]    
                # Iterate in a 3x3 neighborhood centered at (x, y)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        new_x, new_y = x + dx, y + dy            
                        # Skip out-of-bounds or central cell
                        if (0 <= new_x < imgLen) and (0 <= new_y < imgLen) and (dx != 0 or dy != 0):
                            if sstnp_array[new_x, new_y] <= 10:
                                sstnp_array[x, y] = sstnp_array[new_x, new_y]
                                break
            
            sst_tensor=torch.from_numpy(sstnp_array)            
            y=item['Ydata']      
            truthT=[] 
            lenitem=45
            for cur in range(0,lenitem):
                indArgo=indexArgo[cur]         
                truthT.append(y[indArgo])    
                
            tcx_item_seq=torch.tensor([[x1_year,x2_diffday,x3_argo_lon,x4_argo_lat,x5_eddy_radius,x6_eddy_eke,x7_eddy_amp,x8_eddy_nd]])
            tcy_item=torch.tensor(truthT)              
            sample_data_item={'sst_image':sst_tensor,'sequence':tcx_item_seq,'youtput':tcy_item}  
            iterflat=iterflat+1            
            if(ae_firstFlag==1):
                ae_ds_X=[sample_data_item]
                ae_ds_Y=tcy_item
                ae_firstFlag=0
            else:
                ae_ds_X.append(sample_data_item)
                ae_ds_Y=torch.cat([ae_ds_Y,tcy_item],axis=0)
            
            
#train ae and ce seperately 
          


# normalise the data is need, method could be chosen accoring to your situation 
# Normalise the input data
update_ae_ds_X=[]
train_dataset,test_dataset=random_split(dataset=update_ae_ds_X,lengths=[35230,9936],generator=torch.Generator().manual_seed(0))

import math
# Create an instance of CustomDataset
Train_dataset = CustomDataset(train_dataset)
Test_dataset = CustomDataset(test_dataset)
# Create a DataLoader for batching and shuffling
Train_dataloader = DataLoader(Train_dataset, batch_size=1, shuffle=True)
Test_dataloader= DataLoader(dataset=Test_dataset,batch_size=1)

# Create an instance of the SPPNet
net = SPPNet_s()

# Define a loss function and optimizer
optimizer=torch.optim.Adam(net.parameters(),lr=xxx)
loss_func=nn.MSELoss() #
criterion = nn.MSELoss()
mae_loss = nn.L1Loss()

# simple example to train the network
for epoch in range(yy):  # Adjust the number of epochs as needed
    step=0    
    for data in Train_dataloader:
        image_data = data['sst_image']
        image_tensor = Variable(torch.unsqueeze(image_data, dim=0).float(), requires_grad=False)
        sequence_data = data['sequence'].unsqueeze(0)  # Add batch dimension
        target=data['youtput'].unsqueeze(0)        
        # Forward pass        
        output = net(image_tensor, sequence_data)         
        # Compute the loss
        loss = criterion(output, target)         
        train_loss_all.append(loss) # record loss        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    for t_data in Test_dataloader:
        t_image_data = t_data['sst_image']
        t_image_tensor = Variable(torch.unsqueeze(t_image_data, dim=0).float(), requires_grad=False)
        t_sequence_data = t_data['sequence'].unsqueeze(0)  # Add batch dimension
        t_target=t_data['youtput'].unsqueeze(0)
        
        # Forward pass        
        t_output = net(t_image_tensor, t_sequence_data) 
        test_loss=loss_func(t_output,t_target)
        test_loss_epoch.append(test_loss.item())
        testout.append(t_output)

# After training, you can use the trained network for predictions
    