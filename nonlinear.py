# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:02:45 2019

@author: LENOVO
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import inv
#%matplotlib auto
m=np.zeros(3)
m[0]=1.5
m[1]=1.5
m[-1]=1.2
t=np.linspace(0,1,100)
E=15
# initial model
m0=np.zeros(3)
m0[0]=1
m0[1]=1
m0[-1]=1
dm1=np.zeros_like(t)
dm2=np.zeros_like(t)
dm3=np.zeros_like(t)
## function definition#############################
def sint(m1,m2,m3,t):
    dsin=m1*np.exp(m2*t+m3*t**2)
    return dsin
def obsm(m1,m2,m3,t):
    dobs=m1*np.exp(m2*t+m3*t**2)
    return dobs

def jac(m1,m2,m3,t):
    dm1=np.exp(m2*t+m3*t**2)
    dm2=m1*t*np.exp(m2*t+m3*t**2)
    dm2=m1*(t**2)*np.exp(m2*t+m3*t**2)
    return dm1,dm2,dm3
def JT(dm1,dm2,dm3):
    J=np.vstack([dm1,dm2,dm3]).T
    return J
def LSQR(J,IB,dd):
    mm=np.dot((inv(np.dot(J.T,J)+IB)),(np.dot(J.T,dd)))
    return mm
def mupdate(mm,m0):
    m0[0]=mm[0]+m0[0]
    m0[1]=mm[1]+m0[1]
    m0[-1]=mm[-1]+m0[-1]
    return m0
    
####################################################
etol=0.6
n=0
NEE=np.zeros(10)
#while E>etol:
while E>etol:
    dsin=sint(m[0],m[1],m[-1],t)
    dobs=obsm(m0[0],m0[1],m0[-1],t)
    dm1,dm2,dm3=jac(m0[0],m0[1],m0[-1],t)
    #J=np.vstack([dm1,dm2,dm3]).T
    J=JT(dm1,dm2,dm3)
    dd=dsin-dobs
    I=np.identity(len(np.dot(J.T,J)))
    IB=I*1e-15
    mm=LSQR(J,IB,dd)
    #mm=np.dot((inv(np.dot(J.T,J)+IB)),(np.dot(J.T,dd)))
    
    #m0[0]=mm[0]+m0[0]
    #m0[1]=mm[1]+m0[1]
    #m0[-1]=mm[-1]+m0[-1]
    m0=mupdate(mm,m0)
    E=np.dot(dd.T,dd)
    
    NEE[n]=E
    n+=1
    #if n>50:
    #    break
    #if n % 1 == 0:
     #   plt.plot(t,dsin,'b',t,dobs)
      #  plt.grid(True)
    
    plt.plot(t,dsin,'b',t,dobs)
    plt.grid(True)
    plt.draw()
    plt.pause(1.5)

plt.figure(2)      
plt.plot(t,dsin,'b',t,dobs,'r')
plt.grid(True)
plt.figure(3)      
plt.plot(NEE,'b')
plt.grid(True)








#while E>2:
#    dsin=