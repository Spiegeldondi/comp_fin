import numpy as np
import matplotlib.pyplot as plt

S = 1
T = 5
m = 100
dt = T/m 
"""
ex1 task:
2*kappa*theta>sigma**2
from ex1:
kappa*dt > 1
kappa*theta ≈ 0
"""
kappa = 1.5/dt
theta = 1/(1e3*kappa)
sigma = 0.5*np.sqrt(2)*np.sqrt(theta*kappa)

# kappa = 100/dt
# theta = 1/(1e1*kappa)
# sigma = 0.5*np.sqrt(2)*np.sqrt(theta*kappa)

print("kappa:", kappa)
print("theta:", theta)
print("sigma:", sigma)

delta = (4*kappa*theta)/sigma**2

def cbar(t, s):
    return sigma**2/(4*kappa)*(1-np.exp(-kappa*(t-s)))

def kbar(t, s, S):
    nom = 4*kappa*np.exp(-kappa*(t-s))*S
    denom = sigma**2*(1-np.exp(-kappa*(t-s)))
    return nom/denom

def EulerMaruyama(St, sigma, dWt):
    return St*(1+r*dt+sigma*dWt)
    
if __name__ == "__main__":
    s = 0
    t = s+dt
    Sdata = np.zeros(m)
    print("delta", delta)
    print("cbar", cbar(t,s))
    print("kbar", kbar(t,s,S))
    for i in range(m-1):
        Sdata[i] = S
        S = cbar(t, s)*np.random.noncentral_chisquare(delta, kbar(t, s, S))
        # print(S)
        s += dt
        t = s+dt
        
    plt.plot(np.linspace(0,T,m),Sdata)
    plt.xlabel("t")
    plt.title(f"CIR with {m} steps")
    plt.ylabel("S")
    plt.show()