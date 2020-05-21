# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 02:17:35 2019

@author: LENOVO
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage
sigma_y = 3.0
sigma_x = 2.0
#x = np.fromfile('marmhard.dat')
data = np.loadtxt( 'marmhard.dat' )
nx=384
ny=122
nxx=nx+1
nyy=ny+1
vp_marm=np.ones([ny,nx])

k=0
for i in range(nx):
    for j in range(ny):
        vp_marm[j,i]=data[k]
        k=k+1

print(data.shape,vp_marm.shape)
plt.figure()
plt.imshow(vp_marm)    
plt.show()    
plt.figure()
plt.imshow(vp_marm, interpolation='nearest')
plt.show()
#vp_marmSM = ndimage.gaussian_filter(vp_marm, sigma=(5, 5, 0), order=0)
# Note the 0 sigma for the last axis, we don't wan't to blurr the color planes together!
sigma = [sigma_y, sigma_x]
vp_smooth = sp.ndimage.filters.gaussian_filter(vp_marm, sigma, mode='nearest')
plt.figure()
plt.imshow(vp_smooth)    
#plt.colorbar()
plt.show()    


















