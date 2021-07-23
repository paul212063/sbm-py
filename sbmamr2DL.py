#SBM for 2D Laplace equation 
#Copyright (C) 2020 Zhuojia FU, Hohai University
import numpy as np
import math
from scipy.special import *
from scipy.spatial.distance import cdist
from scipy import linalg
import timeit

nknot=np.array([100,400,1000,4000,10000])
error1=np.zeros([1, len(nknot)])
cputime=np.zeros([1, len(nknot)])
for itt in range(0,len(nknot)):
    Ra=1.0
    ntknot=nknot[itt]
    the=np.linspace(0.0,2*np.pi-2*np.pi/ntknot,ntknot)
    the.shape=(ntknot,1)
    meansA=2*np.pi*Ra/ntknot
    diagAU=-1.0/2.0/np.pi*(np.log(meansA/2/np.pi))
    Axy=np.column_stack((Ra*np.cos(the), Ra*np.sin(the)))
    distA=np.linalg.norm(Axy[0,:]-Axy,axis=1,keepdims=True)
    distA[0]=1
    Alog=-1.0/2.0/np.pi*(np.log(distA))
    Alog[0] = diagAU
    theta0=0.0
    bxy=Axy[:,0]**2-Axy[:,1]**2
    start = timeit.default_timer()
    rbfcoeff =np.fft.fft(np.fft.ifft(bxy)/np.fft.fft(Alog.T))
    rbfcoeff.shape=(ntknot,1)
    cputime[0,itt] = (timeit.default_timer() - start)
    rho=0.8*Ra
    nnknot=201
    theta=np.linspace(0.0,2*np.pi-2*np.pi/nnknot,nnknot)
    theta.shape=(nnknot,1)
    sAxy=np.column_stack((rho*np.cos(theta), rho*np.sin(theta)))
    ff=sAxy[:,0]**2-sAxy[:,1]**2
    DMT = cdist(sAxy,Axy,metric='euclidean')
    DMFS=-1.0/2.0/np.pi*(np.log(DMT))
    fZI=np.dot(DMFS,rbfcoeff)
    error1[0,itt]=np.linalg.norm(fZI.T-ff)/np.linalg.norm(ff)



