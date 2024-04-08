  # !/usr/bin/env Python
  # coding=utf-8

"""
Created on Thu Sep 21 13:50:05 2023
map the eddies and the Argo Profile in area
@author: Cathy Duan
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import os
from matplotlib.path import Path
import netCDF4 as nc
from tool_duan import date_from_days_since_1950,date_getStartIndex,date_getEndIndex,getNDval,getRadInfo
import math
import json
from math import cos,pi,sqrt

# select Argo within target region.
for pathitem in os.listdir(ArgoPathHeader): # loop folder
    print (pathitem)
    fullpathName=os.path.join(ArgoPathHeader,pathitem)      
    argo_datainArea_PerMonth={}
    for mfileitem in os.listdir(fullpathName): #loop file in the folder
        print (mfileitem)
        MfullpathName=os.path.join(fullpathName,mfileitem) 
        envVars=readArgoDataAll(MfullpathName)
        if(len(envVars)>0):
            slat=envVars['LATITUDE']
            slon=envVars['LONGITUDE']
            if (slat>5.) and (slat<60.) and (slon>105.) and (slon<180.):
                print ('found one:'+str(slon)+','+ str(slat))
                sdate=envVars['date'].replace('-','')
                    #skey=sdate+'_'+str(slat)+'_'+str(slon)
                    #argo_datainArea_PerMonth[skey]=envVars
                if sdate in argo_datainArea_PerMonth:
                    argo_datainArea_PerMonth[sdate].append(envVars)
                else:
                    argo_datainArea_PerMonth[sdate]=[]
                    argo_datainArea_PerMonth[sdate].append(envVars)
                
                    
    savePerMonth= ArgoSavePath + r'xxx.json'  #save the Argo in region
    with open(savePerMonth,'w',encoding='utf-8') as fp:
        json.dump(argo_datainArea_PerMonth,fp)
      
#mapping
logstring=[]
for pathitem in os.listdir(ArgoPathHeader): # loop files in folder per month
    argo_datainArea_PerMonth={}
    fullpathName=os.path.join(ArgoPathHeader,pathitem) 
    with open(fullpathName,'r') as f:
        argo_datainArea_PerMonth=json.load(f) #read dict， the key is daily date， there are several profile in each day
    NumArgo=0
    NumfoundEddy=0
    argo_withEddy_PerMonth={}
    for date_ in argo_datainArea_PerMonth:
        oneday_argo=argo_datainArea_PerMonth[date_]        
        for dateitem in eddys:
            if(dateitem!=date_):
                continue
            onedaylist=eddys[dateitem] #find oneday list and break
            break
        for oneargo in oneday_argo:
            NumArgo=NumArgo+1
            flagFound=0
            lons_eddy=[]
            lats_eddy=[]
            lon_argo=oneargo['LONGITUDE']
            lat_argo=oneargo['LATITUDE'] 
            for oneEddy in onedaylist: 
                #calc in unified unit 
                rad=oneEddy['x5_eddy_radius']/1000
                c_lon=oneEddy['x3_argo_lon']
                c_lat=oneEddy['x4_argo_lat']
                rad_lat,rad_lon=getRadInfo(rad,c_lat)
                if lon_argo>c_lon-rad_lon and lon_argo<c_lon+rad_lon and lat_argo>c_lat-rad_lat and lat_argo<c_lat+rad_lat:
                    ndval=getNDval(lon_argo,lat_argo,c_lon,c_lat)                    
                    if ndval<=1:                      
                        oneargo['x3_argo_lon']=c_lon
                        oneargo['x4_argo_lat']=c_lat
                        oneargo['xothers_cost']=oneEddy['xothers_cost']
                        oneargo['xothers_sint']=oneEddy['xothers_sint']
                        oneargo['xothers_cos2t']=oneEddy['xothers_cos2t']
                        oneargo['xothers_sin2t']=oneEddy['xothers_sin2t']                            
                        oneargo['x5_eddy_radius']=rad
                        oneargo['x6_eddy_eke']=oneEddy['x6_eddy_eke']
                        oneargo['x7_eddy_amp']=oneEddy['x7_eddy_amp']
                        oneargo['x8_eddy_nd']=ndval
                        oneargo['xothers_eddy_contour_lon']=oneEddy['xothers_eddy_contour_lon']
                        oneargo['xothers_eddy_contour_lat']=oneEddy['xothers_eddy_contour_lat']
                           
                        flagFound=1
                        NumfoundEddy=NumfoundEddy+1  
                        break
                if(flagFound==1):
                    break
                
            if(flagFound==1): 
                #matched one eddy，save to dict
                if date_ in argo_withEddy_PerMonth:
                    argo_withEddy_PerMonth[date_].append(oneargo)
                else:
                    argo_withEddy_PerMonth[date_]=[]
                    argo_withEddy_PerMonth[date_].append(oneargo)            

    onestring='xxxx\n'
    logstring.append(onestring)
  #save to file
