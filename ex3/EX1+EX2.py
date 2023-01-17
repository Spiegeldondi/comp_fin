#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:04:07 2022

@author: alex
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

nasdaq = pd.read_csv('/home/dominik/cse/2022W/finance/ex3/HistoricalPrices_NASDAQ.csv',sep=',;')

nasdaq = nasdaq.loc[::-1] #reverse order
nasdaq = nasdaq.reset_index(drop = True)

#nasdaq.drop('Volume',axis=1,inplace = True) # drop volume column

close = nasdaq['Close']
#close_np = close.to_numpy()

close_X = np.log(close/close.iloc[0])
n = len(close)+1
T = len(close)
dt = T/n
m = len(close)

mu_estimate = (close_X.iloc[-1] - close_X.iloc[0]) / (m*dt) # estimate for drift

x0 = close_X.iloc[0:m-1]
x1 = close_X.iloc[1:m].reset_index(drop = True) # dropping index so it starts with 1 again
dx = x1 - x0

sigma_estimate = 1/m * sum(((dx-mu_estimate*dt)**2)/dt) # estimate for squared volatility

def W_t(T,n):
    dt = T/n
    x = np.zeros(n)
    for i in range(n):
        x[i] = x[i-1] + np.random.normal(0,dt)
    return x


s0 = close.iloc[0]
s_t = np.zeros(n)
t = np.linspace(0,T,n)
s_t = s0*np.exp(mu_estimate*t + np.sqrt(sigma_estimate)*W_t(T,n))

plt.figure(0)
plt.plot(t,s_t)
plt.ylabel("S_t")

plt.figure(1)
plt.plot(close)
plt.ylabel("close")






