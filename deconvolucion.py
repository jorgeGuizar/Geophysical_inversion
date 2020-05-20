# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 02:43:26 2019

@author: LENOVO
"""
"""
import numpy as np
 Given: x(t)
% Find inverse of x(t) by spectral deconvolution
%  dt     -input- Temporal sampling interval
%  eps0   -input- Damping parameter
%  fr     -input- Dominant frequency of Ricker
%  np     -input- # of samples for input wavelet
%  x(t)   -input- Ricker wavelet
%  IX(f)  -output- Spectral decon filter
%  ix(t)  -output- Result after decon filter applied to x(t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
dt=.002
fr=17
nsamples=1024#for FFT
T=np.round(1/20/dt)
npt=nsamples*dt
t=np.arange(-npt/2,npt/2,dt)
# input vwavlet
x=(1-t*t*(fr**2) *(np.pi**2))*np.exp(- t**2 *np.pi**2 * fr**2 ) 
nt=len(x)
A=int(np.round(nt/2))-int(T)
B=int(nt-1)
xnew=x[A:B]
nt=len(xnew)
tiempo=np.arange(0,nt)*dt
plt.figure()
plt.subplot(2,2,1)
plt.plot(tiempo,xnew)

nt2=np.round(nt/2)
df=1/((nt-1)*dt)
X=fft(xnew)
tiempo2=np.arange(0,nt2)*dt
FFTamp=np.abs(X[0:int(nt2)])


plt.subplot(2,2,2)
plt.plot(tiempo2,FFTamp)

eps0=0.05
eps1=np.max(np.abs(X**.2))*eps0
IX=np.conj(X)/(np.conj(X)*X+eps1)

freq=np.arange(0,nt2)*df
FFTamp2=np.abs(IX[0:int(nt2)])
plt.subplot(2,2,4)
plt.plot(freq,FFTamp2)

ix=ifft(IX*X)
ntt=len(ix)
ntt2=np.round(ntt/2)
tiempo3=np.arange(-ntt2,ntt2-1)*dt
M=(ix[int(ntt2):ntt])
N=ix[0:int(ntt2-1)]
coef=np.block([M,N])
plt.subplot(2,2,3)
plt.plot(tiempo3,np.real(coef))
#ixx=[np.flipud(ix[int(nt2):int(nt)]),ix[0:int(nt2)]]
#plt.plot(np.arange(nt,nt2-1)*dt,np.real(ixx))






















