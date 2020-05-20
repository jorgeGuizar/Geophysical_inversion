# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:35:41 2019

@author: jorge Guizar alfaro
ETH zurich
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import inv
#%matplotlib auto
###############################################################################
plt.close()
m=np.zeros(3)
Z=50 #depth in meters
R=20 # radius of sphere in m
RHO1=2 # density of medium 1
RHO2=3.5 #density of medium 2
DRHO=RHO2-RHO1 #density contrast
m[0]=R
m[1]=Z
m[-1]=DRHO
x=np.linspace(-200,200,200)
G=6.67408e-11 
###############################################################################
# initial model for the iterations
m0=np.zeros(3)
Z0=40 #depth in meters
R0=15 # radius of sphere in m
DRHO_0=-1.5 #density contrast
m0[0]=R0
m0[1]=Z0
m0[-1]=DRHO_0
##############################################################################
#initialize  jacobian vectors
dR=np.zeros_like(x)
dZ=np.zeros_like(x)
dRHO=np.zeros_like(x)
## function definition########################################################
def sintgrav(R,Z,DRHO,x):
    dsin=(4/3)*np.pi*(R**3)*DRHO*G*(Z/(x[:]**2+Z**2)**(3/2))
    return dsin
def obsgrav(R0,Z0,DRHO_0,x):
    dobs=(4/3)*np.pi*(R0**3)*DRHO_0*G*(Z0/(x[:]**2+Z0**2)**(3/2))
    return dobs

def jac(R0,Z0,DRHO_0,x):
    dR=4*np.pi*(R0**2)*DRHO_0*G*(Z0/(x[:]**2+Z0**2)**(3/2))#dR
    dff=(((x[:]**2+Z0**2)**(3/2)-((Z**2)*3*(x[:]**2+Z0**2)**(1/2)))/(x[:]**2+Z0**2)**3)
    dZ=(4/3)*np.pi*(R0**3)*DRHO_0*G*dff#dZ
    dRHO=(4/3)*np.pi*(R0**3)*G*(Z0/(x[:]**2+Z0**2)**(3/2))#drho
    return dR,dZ,dRHO

def JT(dR,dZ,dRHO):
    J=np.vstack([dR,dZ,dRHO]).T
    return J
def LSQR(J,IB,dd):
    mm=np.dot((inv(np.dot(J.T,J)+IB)),(np.dot(J.T,dd)))
    return mm
def mupdate(mm,m0):
    m0[0]=mm[0]+m0[0]
    m0[1]=mm[1]+m0[1]
    m0[-1]=mm[-1]+m0[-1]
    return m0
##############################################################################
E=15
etol=1e-20
n=0
NEE=np.zeros(10)
#while E>etol:
while E>etol:
    dsin=sintgrav(m[0],m[1],m[-1],x)
    dobs=obsgrav(m0[0],m0[1],m0[-1],x)
    dR,dZ,dRHO=jac(m0[0],m0[1],m0[-1],x)
    J=JT(dR,dZ,dRHO)
    dd=dsin-dobs
    I=np.identity(len(np.dot(J.T,J)))
    IB=I*1e-20
    mm=LSQR(J,IB,dd)
    m0=mupdate(mm,m0)
    E=np.dot(dd.T,dd)
    NEE[n]=E
    n+=1
    #if n>50:
    #    break
    #if n % 1 == 0:
     #   plt.plot(t,dsin,'b',t,dobs)
      #  plt.grid(True)
    plt.plot(x,dsin,'b')
    plt.plot(x,dobs,label='%s step inversion' % n)
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(1)
    #E=1e-12

plt.figure(2)      
plt.plot(x,dsin,'b',label='synthetic')
plt.plot(x,dobs,'r',label='inverted')
plt.legend()
plt.grid(True)
plt.figure(3)      
plt.plot(NEE,'b')
plt.grid(True)