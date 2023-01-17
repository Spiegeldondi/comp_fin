import numpy as np
import matplotlib.pyplot as plt

"""
ex1 task:
2*kappa*theta>sigma**2
from ex1:
kappa*dt > 1
kappa*theta ≈ 0
"""
kappa = 10
theta = 10
sigma = np.sqrt(2*kappa*theta)+100

# kappa = 1.5/dt
# theta = 1/(1e3*kappa)
# sigma = 0.5*np.sqrt(2)*np.sqrt(theta*kappa)

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
    nom = 4*kappa*np.exp(-kappa*(t-s))
    denom = sigma**2*(1-np.exp(-kappa*(t-s)))
    return (nom/denom)*S
    
def runSim(S0, dt, m):
    s = 0
    t = s+dt
    Sdata = np.zeros(m)
    S = S0
    for i in range(m-1):
        Sdata[i] = S
        S = cbar(t, s)*np.random.noncentral_chisquare(delta, kbar(t, s, S))
        # print(S)
        s += dt
        t = s+dt
    return Sdata

if __name__ == "__main__":
    T = 1500
    S0 = 1
    dt1 = 0.5
    dt2 = 0.1
    dt3 = 0.05
    data1 = runSim(S0,dt1,int(np.ceil(T/dt1)))
    data2 = runSim(S0,dt2,int(np.ceil(T/dt2)))
    data3 = runSim(S0,dt3,int(np.ceil(T/dt3)))
        
    plt.plot(np.linspace(0,T,int(np.ceil(T/dt1))),data1, label=dt1, alpha=0.7)
    plt.plot(np.linspace(0,T,int(np.ceil(T/dt2))),data2, label=dt2, alpha=0.7)
    plt.plot(np.linspace(0,T,int(np.ceil(T/dt3))),data3, label=dt3, alpha=0.7)

    plt.legend()
    plt.title(f"CIR")
    plt.ylabel("S")
    plt.xlabel("t")
    plt.show()