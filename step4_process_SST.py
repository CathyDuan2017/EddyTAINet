# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:30:12 2023
@author:  Yingying Duan
"""

from scipy import signal
import numpy as np
import numpy.ma as ma

from readnc import changeNCDatabyLatIndex,getOneLatAllLonfromCutFile

#step1 cut file per day
import os
import netCDF4 as nc
from tool_duan import cutSSTofR09,write_ncbyData,getRefCutData
import matplotlib.pylab as plt
import scipy.ndimage as ndimage



file20YearsList=['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011',
                 '2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022'] #filepath name of 20 years files
pathHeader=r'x:\SST\\'
for item1 in file20YearsList:    
    i=0
    filepath=pathHeader+item1
    for item in os.listdir(filepath):    #
        fullpathName=os.path.join(filepath,item)
        CutFileName=savePathHeader+item
        file_obj = nc.Dataset(fullpathName)     
        
        [cut_data,startLat,endLat,startLon,endLon]=cutSSTofR09(file_obj)
        write_ncbyData(CutFileName,file_obj,cut_data,startLat,endLat,startLon,endLon)
        file_obj.close()
        

# temporal filter
[refData,latdata,londata,startLat,endLat,startLon,endLon]=getRefCutData()
fillvalue=refData.fill_value
[latSize,lonSize]=refData.shape
for latIndex in range(0,latSize):
    matrxCol=[]
    matrxCol_msk=[]
    matrFilterCol=[]
    
    firstFlag=True
    for item in os.listdir(savePathHeader):
        fullpathName=os.path.join(savePathHeader,item) 
        columData=getOneLatAllLonfromCutFile(fullpathName,latIndex)
        if(firstFlag==True):
            matrxCol=columData.data
            matrxCol_msk=columData.mask
            firstFlag=False
        else:
            matrxCol=np.c_[matrxCol,columData] 
            matrxCol_msk=np.c_[matrxCol_msk,columData.mask] 
    matrxCol=matrxCol.transpose() 
    matrxCol_msk=matrxCol_msk.transpose()
    maskMattCol=ma.masked_array(matrxCol,mask=matrxCol_msk)
    for lonIndex in range(0,lonSize):
        matrFilterCol=tempFiter(maskMattCol[:,lonIndex])

    indFile=0
    for item in os.listdir(savePathHeader):
        sstColData=matrFilterCol[indFile][:]
        fullpathName=os.path.join(savePathHeader,item)
        changeNCDatabyLatIndex(fullpathName,latdata,londata,latIndex,sstColData)
        indFile+=1
    if( latIndex % 10 == 0):    
        del matrxCol
        del matrFilterCol
        del matrxCol_msk
#        del matrFilterColWithM
        import gc
        gc.collect()                    
                    


         




