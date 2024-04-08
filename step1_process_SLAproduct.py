Created on Thu Sep 21 13:50:05 2023
Process the mesoscale eddy trjectories atlas product
@author: Cathy Duan

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import os
from matplotlib.path import Path
import netCDF4 as nc
from tool_duan import date_getStartIndex,date_getEndIndex
import math
import json

file_ae=r'x:\Aviso\META3.2_DT_allsat_Anticyclonic_long.nc' # can be downloaded and chosen from AVISO+
file_ce=r'x:\Aviso\META3.2_DT_allsat_Cyclonic_long.nc'     # can be downloaded and chosen from AVISO+
# process AEs 
fileAE_obj = nc.Dataset(file_ae)
#cut the data within range of the following:
#1 time range 2002.6.1~2022.5.31 20 years
#2 lat range 5N-60N
#3 lon range105-180E
ae_datelist=fileAE_obj.variables['time'][:]  #([startDayNo,EndDayNo])
ae_latlist=fileAE_obj.variables['latitude'][:]
ae_lonlist=fileAE_obj.variables['longitude'][:]

ae_amp_list=fileAE_obj.variables['amplitude'][:]
ae_effArea_list=fileAE_obj.variables['effective_area'][:]
ae_effConLat_list=fileAE_obj.variables['effective_contour_latitude'][:]
ae_effConLon_list=fileAE_obj.variables['effective_contour_longitude'][:]
ae_effRadius_list=fileAE_obj.variables['effective_radius'][:]
ae_speed_Radius_list=fileAE_obj.variables['speed_radius'][:]
ae_speedAverage_list=fileAE_obj.variables['speed_average'][:]

AE_eddyCutData={}
numEddy=0
MaxNumber=ae_datelist.shape[0]
for i in range(MaxNumber):
    cur=i    
    itemLat=ae_latlist[i]
    itemLon=ae_lonlist[i]
    dateDiff=ae_datelist[i]
    if (itemLat<5) or (itemLat >60):
        continue
    if (itemLon<105) or (itemLon>180):
        continue
    if (dateDiff<dateStart) or (dateDiff>dateEnd):
        continue  
    oneitem={}
    curdate_dtType=date_from_days_since_1950(dateDiff)
    curyear=curdate_dtType.year
    curdate_str=curdate_dtType.strftime('%Y%m%d')    
    oneitem['x1_year']=curyear
    oneitem['x2_diffday']=dateDiff
    oneitem['x3_argo_lon']=float(itemLon)
    oneitem['x4_argo_lat']=float(itemLat)    
    estimateKE=(ae_speedAverage_list[i]/2)*ae_speedAverage_list[i]
    estimateKE=estimateKE/N    
    oneitem['x5_eddy_radius']=ae_effRadius_list[i]
    oneitem['x6_eddy_eke']=estimateKE
    oneitem['x7_eddy_amp']=ae_amp_list[i]
    oneitem['x8_eddy_nd']=1  # will calculate again combined with Argo float location.
    oneitem['xothers_eddy_contour_lon']=ae_effConLon_list[i].tolist()
    oneitem['xothers_eddy_contour_lat']=ae_effConLat_list[i].tolist()
    oneitem['Ydata']=[]  # will update according to Argo TA profiles
    #save 
    if curdate_str in AE_eddyCutData:
        AE_eddyCutData[curdate_str].append(oneitem)                        
    else:
        AE_eddyCutData[curdate_str]=[]
        AE_eddyCutData[curdate_str].append(oneitem) 
        numEddy=numEddy+1

