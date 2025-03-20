#!/usr/bin/env python
# coding: utf-8

# In[ ]:


######Sample###########
#read tiff file and create array

import tifffile
import numpy as np

#Input directory name include tiff_file (N=800)
#Output xaxis, yaxis, 3D datacube
def tiffr(path):
    file0 = '0001.tif'
    image = tifffile.imread(path+file0)
    nx0, ny0 = image.shape[1], image.shape[0]
    x0 = np.arange(0,nx0,1)
    y0 = np.arange(0,ny0,1)
    Z = np.zeros((800,ny0,nx0))

    #create 3D data cube sample
    for i in range(800):
        Num = i+1
        if Num < 10:
            file = '000'+str(Num)+'.tif'
        elif Num < 100:
            file = '00'+str(Num)+'.tif'
        else:
            file = '0'+str(Num)+'.tif'

        image1 = tifffile.imread(path+file,key=0)
        Z[i,:,:] = image1
        
    return x0,y0,Z

