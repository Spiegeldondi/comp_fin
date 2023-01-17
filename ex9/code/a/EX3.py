#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:14:15 2023

@author: alex
"""
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

def stoch_int(f,T,dt = 'stepsize' ,N = 'Number of MC samples'):
    n = int(np.ceil(T/dt))
    t = np.linspace(0,T,n)
    integral = np.zeros(N) 
    for j in range(N):
        for i in range(n-1):
            integral[j] += f((t[i+1]+t[i])/2)*normal(0,dt) 
    
    return integral  
    

I1 = stoch_int(lambda x: np.cos(x),2,0.1,1000)

plt.figure()
plt.title('cos(t)')
plt.hist(normal(0,1/4*(4+np.sin(4)),1000),  bins = 25, label = 'Analyticaly', density = True)
plt.hist(I1, bins = 25, label = 'MC', density = True)
plt.legend()

'''I2 = stoch_int(lambda x: np.exp(x),2,0.1,1000)
plt.title('exp(t)')
plt.hist(I2, bins = 25, label = 'MC')
plt.hist(normal(0,1/4*(4+np.sin(4)),1000),  bins = 50, label = 'Analyticaly')
plt.legend()'''


