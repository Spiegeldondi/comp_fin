#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:14:15 2023

@author: alex
"""
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from scipy import stats

def stoch_int(f,T,dt = 'stepsize' ,N = 'Number of MC samples'):
    n = int(np.ceil(T/dt))
    t = np.linspace(0,T,n)
    integral = np.zeros(N) 
    for j in range(N):
        for i in range(n-1):
            integral[j] += f((t[i+1]+t[i])/2)*normal(0,dt) 
    
    return 1/N*np.sum(integral)   
    

I1 = stoch_int(lambda x: np.cos(x),2,0.1,1000)
print(f' Das Integral für f = cos(t) und T = 2 : {I1}')       

I2 = stoch_int(lambda x: np.exp(x),2,0.1,1000)
print(f' Das Integral für f = exp(t) und T = 2 : {I2}')       

data = []
for i in range(0,100):
    data.append(stoch_int(lambda x: np.cos(x),2,0.1,1000))

print(np.mean(data), np.var(data))
plt.hist(data)
x = np.linspace(-0.5,0.5,100)
plt.plot(x, stats.norm.pdf(x, loc=0, scale=0.8108)*100)
plt.show()
