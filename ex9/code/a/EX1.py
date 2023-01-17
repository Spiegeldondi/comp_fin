#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 02:02:11 2023

@author: alex
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


S_n = np.linspace(0,1500,100)
k = 10
theta = 10
sigma = np.sqrt(2*k*theta)+100


plt.figure()
dt = 0.5
plt.plot(S_n,norm.cdf(-(S_n + k*(theta - S_n)*dt)/(sigma*np.sqrt(S_n*dt))), label = "dt = 0.5")

dt = 0.1
plt.plot(S_n,norm.cdf(-(S_n + k*(theta - S_n)*dt)/(sigma*np.sqrt(S_n*dt))), label = 'dt = 0.1')

dt = 0.01
plt.plot(S_n,norm.cdf(-(S_n + k*(theta - S_n)*dt)/(sigma*np.sqrt(S_n*dt))), label = 'dt = 0.01')

plt.title('k = 10; theta = 10; sigma = np.sqrt(2*k*theta)')

#plt.xlabel('S_n')
plt.legend()

