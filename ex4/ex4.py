import numpy as np
import matplotlib.pyplot as plt

r = 0.02
sigma = 0.07
T = 1
K = 12
S = 15

def d2(S, t):
    return (np.log(S/K) + (r-sigma**2/2)*(T-t)) / (sigma * np.sqrt(T-t))

def phiPrime(d):
    return 1/np.sqrt(2*np.pi)*np.exp(-(d**2)/2)

def delta(S, t):
    return K*np.exp(-r*(T-t))*phiPrime(d2(S, t))*1/(sigma*np.sqrt(T-t))*1/S

x = np.linspace(0,0.99,10)
data = delta(S,x)

plt.plot(x,data)
plt.xlabel("t")
plt.ylabel("Δ") # Δ ∂V/∂S
plt.title("Delta of Put Cash or Nothing")
plt.show()
