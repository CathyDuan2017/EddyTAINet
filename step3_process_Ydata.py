  # !/usr/bin/env Python
  # coding=utf-8

"""
Created on Thu Oct 7 13:50:05 2023
Process the y (TA) values, and the salnity values
@author: Cathy Duan
"""

import json
import os
from tool_duan import checkArgo,interplotFun_c,subtractClimate_withIP,getdaysDiff

import matplotlib.pyplot as plt
import netCDF4 as nc


filePath=r'x:\woa18.nc'
woa_obj = nc.Dataset(filePath)

processedData={}
logstring=[]

for mfileitem in os.listdir(AE_pathArgoMatchedEddyt): 
    numerperfile_all=0
    numerperfile_ok=0
    fullFileName=os.path.join(AE_pathArgoMatchedEddyt,mfileitem) 
    with open(fullFileName,'r') as fp:
        eddys=json.load(fp)
        for date_ in eddys:  
            oneDayValues=eddys[date_]
            for item in oneDayValues:
                oneitem={}
                numerperfile_all=numerperfile_all+1
                fDirection=item['direction'] #condition 1
                fQC=item['QC'] #condition 2
                tData=item['TSdata']
                lon_argo=item['LONGITUDE']
                lat_argo=item['LATITUDE']

                if (fQC!='1' and fQC!='2'):
                    continue
                else: #
                    checkFlag=checkArgo(tData) #check the Argo profiles according to rules in paper
                    if checkFlag==False:
                        continue
                    else:  
                        print(date_)
                        pi_Tdata,pi_Depthdata=interplotFun_c(tData)
                        [truthT,woaflag]=subtractClimate_withIP(woa_obj,pi_Tdata,lon_argo,lat_argo)
                                        
                        if(woaflag==False):
                            continue
                        #get input 
                        oneitem['x1_year']=int(date_[0:4])
                        diffdays=getdaysDiff(date_)
                        oneitem['x2_diffday']=diffdays
                        
                        oneitem['eddy_c_lon']= item['x3_argo_lon'] 
                        oneitem['eddy_c_lat']=item['x4_argo_lat']
                        oneitem['x3_argo_lon']=lon_argo  
                        oneitem['x4_argo_lat']=lat_argo
                        oneitem['xothers_cost']=item['xothers_cost']
                        oneitem['xothers_sint']=item['xothers_sint']
                        oneitem['xothers_cos2t']=item['xothers_cos2t']
                        oneitem['xothers_sin2t']=item['xothers_sin2t']
                        
                        oneitem['x5_eddy_radius']=item['x5_eddy_radius']
                        oneitem['x6_eddy_eke']=item['x6_eddy_eke']
                        oneitem['x7_eddy_amp']=item['x7_eddy_amp']
                        oneitem['x8_eddy_nd']=item['x8_eddy_nd']
                        
                        oneitem['Ydata']=truthT
                        
                        #save
                        if date_ in processedData:
                            processedData[date_].append(oneitem)                        
                        else:
                            processedData[date_]=[]
                            processedData[date_].append(oneitem) 
                        numerperfile_ok=numerperfile_ok+1
                        
    logitem="xx\n"
    logstring.append(logitem)                            
#save file
with open(saveAlldataPathAEall,'w',encoding='utf-8') as fp:
    json.dump(processedData,fp) 







