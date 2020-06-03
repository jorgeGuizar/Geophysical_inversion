# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:38:58 2020

@author: LENOVO
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve,inv
from numpy.random import randn

plt.cla()
plt.clf()
plt.close('all')



# 
###################################
def linefit1(xi,ti,flag=True):
    """
    # Function to estimate the parameters of a regression line  
    # using the least-squares method
    #
    # t[i]  = a x[i] + b + e[i]
    # 
        # flag==true   --- >  Estimate only a (assume t=0 for x=0)
        # flag==flase  --- >  Estimate a and b
    #
    # ----------------------------------
    # M D Sacchi
    # GEOPH 326, University of Alberta
    # 2018
    # modified by JGuizar for python
    # ----------------------------------
    """
    # initialize the matrices
    if flag==True:
        a_estimated=np.sum(ti*xi)/np.sum(xi**2)
        b_estimated=0;
    else:
        M=np.zeros([2,2]);
        v=np.zeros(2);
        M[0,0]=np.sum(xi**2)    
        M[0,-1] = np.sum(xi)
        M[-1,0] = np.sum(xi)
        M[-1,-1] = len(xi)*1.0
        
        v[0] = np.sum(ti*xi)
        v[-1] = np.sum(ti)
        u =solve(M,v) 
        a_estimated=u[0]
        b_estimated=u[-1]
    m_est=np.zeros(2)
    m_est[0]=a_estimated
    m_est[-1]=b_estimated
    
    return m_est

#####################################
#####################################

def linefitLS_GN(xi,ti,flag=True):
    """
    # Function to estimate the parameters of a regression line  
    # using the least-squares method
    #
    # t[i]  = a x[i] + b + e[i]
    # 
        # flag==true   --- >  Estimate only a (assume t=0 for x=0)
        # flag==flase  --- >  Estimate a and b
    #
    # ----------------------------------
    # Modified by JGuizar for python
    # ----------------------------------
    """
    # initialize matrices
    #G=np.zeros([len(xi),2])
    if flag==True:
        #G=np.zeros([len(xi)])
        #G=xi.copy()
        #G[:,1]=np.ones(len(xi))
        #GTG=G*G
        #GTD=G*ti
        #m_one=solve(GTG,GTD)
        a_estimated=np.sum(ti*xi)/np.sum(xi**2)
        b_estimated=0;
        m_est=np.zeros(2)
        m_est[0]=a_estimated
        m_est[-1]=b_estimated
        
    else:
        G=np.zeros([len(xi),2])
        G[:,0]=xi
        G[:,1]=np.ones(len(xi))
        GTG=np.dot(G.T,G)
        GTD=np.dot(G.T,ti)
        m_est=np.dot(inv(GTG),GTD)
    return m_est


#################################################
###################################
####################################################(
# Compute synthetic coordinates (xi,ti) and then use them to estimate the slope of 
# of the regression line
xi=np.array([0.22, 1.2, 1.9, 2.3, 3.1, 3.9, 4.1, 5.4])
ti = 1.2*xi + 0.3*randn(len(xi))

m_estimated1 = linefit1(xi,ti,flag=True)
m_estimated2 = linefit1(xi,ti,flag=False)

m_est1 = linefitLS_GN(xi,ti,flag=True)
m_est2 = linefitLS_GN(xi,ti,flag=False)
# vector definition
x2=np.linspace(0,np.max(xi)+1,100)

t1=m_estimated1[0]*x2+m_estimated1[-1]
t2=m_estimated2[0]*x2+m_estimated2[-1]
t3=m_est1[0]*x2+m_est1[-1]
t4=m_est2[0]*x2+m_est2[-1]
# Plot observations and regression line 

plt.figure(1)
plt.plot(xi,ti,'k*',label=r'Original data') 

plt.plot(x2,t3,'g-',label=r'Regression Constrained $b=0$')
plt.plot(x2,t4,'y--o',label=r'Regression No Constrained $b\mathbf{\neq} 0$')
plt.plot(0,0,'oc')
plt.legend(loc='best')
plt.title("Linear Regression LS")
plt.xlabel(r'$x$')
plt.ylabel(r'$T$')
plt.grid(True)
plt.tight_layout()

plt.figure(2)
plt.plot(xi,ti,'k*',label=r'Original data') 
plt.plot(x2,t1,'b--',label=r'Regression Constrained $b=0$')
plt.plot(x2,t2,'r',label=r'Regression No Constrained $b\mathbf{\neq} 0$')

plt.plot(0,0,'oc')
plt.legend(loc='best')
plt.title("Linear Regression LS")
plt.xlabel(r'$x$')
plt.ylabel(r'$T$')
plt.grid(True)
plt.tight_layout()