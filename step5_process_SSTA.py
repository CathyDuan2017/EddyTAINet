# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import json
from scipy.io import savemat
import netCDF4 as nc
import xarray as xr
import numpy as np
import os
import math
from tools import calculate_longitude,calculate_latitude,get_imageInfor
from math import radians, pi, degrees, cos
import cv2

# extract interested information
File_basedCE=r'xx.json'

datalistBasedOrigin={}
newDatalistforSST={}
    
with open(File_basedCE,'r') as fp:
    eddys=json.load(fp)
    for date_ in eddys:  
        oneDayValues=eddys[date_]
        SSTNameHeader = r'x:\SST\xx' 
        NameMid=str(date_)
        NameTail=r'xxx.nc'
        SSTa_file_name=SSTNameHeader+'\\'+NameMid+NameTail
        sstobj=nc.Dataset(SSTa_file_name,'r')
        sstData=sstobj.variables['analysed_sst'][:]  
        sstLat=sstobj.variables['lat'][:]  
        sstLon=sstobj.variables['lon'][:]
        numperdate=0
        for item in oneDayValues:
            numperdate+=1
            newitem={}
            lon_argo=item['eddy_c_lon']
            lat_argo=item['eddy_c_lat']
            rad=item['x5_eddy_radius']
            newitem['date']=date_
            newitem['LONGITUDE']=lon_argo
            newitem['LATITUDE']=lat_argo
            newitem['Radius']=rad
            delta_lat=calculate_latitude(rad)  
            delta_lon=calculate_longitude(lat_argo, rad)
            begin_lat=lat_argo-delta_lat
            end_lat=lat_argo+delta_lat
            begin_lon=lon_argo-delta_lon
            end_lon=lon_argo+delta_lon        
            # lat[5-60],step_delta is about 0.087     
            latCur_begin=math.floor((begin_lat-4.9)/0.1)
            latCur_end=math.ceil((end_lat-4.9)/0.087)
            latInd_begin=latCur_begin
            latInd_end=latCur_end
            if (latCur_begin<0):
                latCur_begin=0            
            if (latCur_end>len(sstLat)):
                latCur_end=len(sstLat)
            foundStart=False
            for lat_ind in range(latCur_begin,latCur_end):
                if (sstLat[lat_ind]>=begin_lat and (not foundStart)):
                    latInd_begin=lat_ind
                    foundStart=True
                if (foundStart and sstLat[lat_ind]>=end_lat):
                    latInd_end=lat_ind
                    break
            
            # lat[5-60],step_delta is about 0.087     
            lonCur_begin=math.floor((begin_lon-104.9)/0.1)
            lonCur_end=math.ceil((end_lon-104.9)/0.087)
            lonInd_begin=lonCur_begin
            lonInd_end=lonCur_end
            if (lonCur_begin<0):
                lonCur_begin=0  
            if (lonCur_end>len(sstLon)):
                lonCur_end=len(sstLon)
            foundStart=False
            for lon_ind in range(lonCur_begin,lonCur_end):
                if (sstLon[lon_ind]>=begin_lon and not foundStart):
                    lonInd_begin=lon_ind
                    foundStart=True
                if (foundStart and sstLon[lon_ind]>=end_lon):
                    lonInd_end=lon_ind
                    break

            cutSST=sstData[latInd_begin:latInd_end,lonInd_begin:lonInd_end]
            restoreSST,length=get_imageInfor(cutSST.data)
            item['SST_lenth']=length
            newitem['SST_lenth']=length
            item['SST_restore']=restoreSST.tolist()
            newitem['SST_restore']=restoreSST.tolist()

            if date_ in datalistBasedOrigin:
                datalistBasedOrigin[date_].append(item)
            else:
                datalistBasedOrigin[date_]=[]
                datalistBasedOrigin[date_].append(item) 

            if date_ in newDatalistforSST:
                newDatalistforSST[date_].append(newitem)
            else:
                newDatalistforSST[date_]=[]
                newDatalistforSST[date_].append(newitem)  
                               



