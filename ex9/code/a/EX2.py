#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:18:54 2023

@author: alex
"""
import numpy as np
from scipy.stats import ncx2
import matplotlib.pyplot as plt


k = 10
theta = 10
sigma = np.sqrt(2*k*theta) + 100
delta = (4*k*theta)/sigma

def c_bar(s,t):
    return sigma**2/(4*k)*(1 - np.exp(-k*(t-s)))

def k_bar(s,t,x0):
    return (((4*k*np.exp(-k*(t-s)))/(sigma**2*(1-np.exp(-k*(t-s)))))*x0)
def S_t_CIR(T,dt,S0):
    n = int(np.ceil(T/dt))
    S = np.zeros(n)
    S[0] = S0
    
    for i in range(n-1):
        S[i+1] = c_bar(i*dt,(i+1)*dt) * ncx2.rvs(delta, k_bar(i*dt,(i+1)*dt,S[i]))
    
    return S

T = 10
dt = 0.1
n = int(np.ceil(T/dt))
t = np.linspace(0,T,n)

plt.figure()
for i in [1,10,100,1000]:
    
    plt.plot(t, S_t_CIR(T,dt,i), label = f'S0 = {i}')
    plt.legend()


'''plt.figure()
for i in [0.01,0.1,0.5]:
    
    plt.plot(np.linspace(0,T,int(np.ceil(T/i))),S_t_CIR(T,i,1), label = f'dt = {i}')
    plt.legend()'''














