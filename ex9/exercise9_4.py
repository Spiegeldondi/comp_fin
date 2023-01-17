import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd

#### ex 4 points I and II

S0 = 1 
T = 2 
r = 0.045 # interest rate
n = 200 # time steps
N = 100000 # number of MCsimulations
a = 0.2
b = -0.3

def sigma(t):
    return a + np.exp(b*t)

def MCsim (S_0, Time, n_steps, rate, func):
    h = Time/n_steps
    S = np.zeros(n_steps)
    S[0] = S_0
    for i in range(n_steps-1):
        S[i+1]=S[i]*(1+rate*h+func(i)*np.sqrt(h)*np.random.normal(0,1)) 
    return S

#K = 1
V0 = []
k = [0.01, 0.1, 0.5, 1, 2, 5]

for K in k:
    print('Strike = ', K)
    for i in range(N):
        s = MCsim (S0, T, n, r, sigma)
        #print('MC iteration: ', i)
        final = s[len(s)-1] - K # call option
        #final = K - s[len(s)-1] # put option
        #print(final)
        payoff = np.zeros(N)
        if final > 0:
            payoff[i] = final

    #print(payoff)
    mean_payoff = np.mean(payoff)
    V0.append(mean_payoff*np.exp(-r*T))




#### ex 4 points III and IV

sigma_match = (1/np.sqrt(2))*(2*a**2 + (2*a/b)*(np.exp(2*b)-1) + (1/(2*b))*(np.exp(4*b)-1))

def d1(S, t):
    return (np.log(S/K) + (r+sigma_match**2/2)*(T-t)) / (sigma_match * np.sqrt(T-t))

def d2(S, t):
    return (np.log(S/K) + (r-sigma_match**2/2)*(T-t)) / (sigma_match * np.sqrt(T-t))

def phi(x):
    return norm(loc=0, scale=1).cdf(x)

def V(S, t, K):
    return S*phi(d1(S, t)) - K*np.exp(-r*(T-t))*phi(d2(S, t))

def W_t(S_0, Time, n_steps):
    dt = Time/n_steps
    x = np.zeros(n_steps)
    x[0] = S_0
    for i in range(1, n_steps):
        x[i] = x[i-1] + np.random.normal(0,dt)
    return x

#S_bs = S0*np.exp((r-(sigma_match**2)/2.)*np.ones(n) + sigma_match*W_t(S0, T, n))

V0_bs = []

for K in k:
    V0_bs.append(V(S0, 0, K))
    #plt.plot(V(S_bs, 0, K))

plt.plot(k, V0, marker='o', label='Monte-Carlo')
plt.plot(k, V0_bs, marker='o', label='Black-Scholes')
plt.show()