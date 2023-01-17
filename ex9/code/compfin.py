import numpy as np
from scipy.stats import norm
T = 2
S0 = 1
r = 0.045
    
def d1(S, sigma, K, t):
    return (np.log(S/K) + (r+sigma**2/2)*(T-t)) / (sigma * np.sqrt(T-t))

def d2(S, sigma, K, t):
    return (np.log(S/K) + (r-sigma**2/2)*(T-t)) / (sigma * np.sqrt(T-t))

def phi(x):
    return norm(loc=0, scale=1).cdf(x)

def V(S, t, K):
    # return S*phi(d1(S, sigma(t), K, t)) - K*np.exp(-r*(T-t))*phi(d2(S, sigma(t), K, t)) # call
    return K*np.exp(-r*(T-t))*phi(-d2(S, sigma(t), K, t)) - S*phi(-d1(S, sigma(t), K, t))

def V_const(S, t, K):
    # return S*phi(d1(S, sigma(t), K, t)) - K*np.exp(-r*(T-t))*phi(d2(S, sigma(t), K, t)) # call
    return K*np.exp(-r*(T-t))*phi(-d2(S, constSigma(t), K, t)) - S*phi(-d1(S, constSigma(t), K, t))

def sigma(t):
    return 0.2+np.exp(-0.3*t)

def constSigma(t):
    x_arr = np.linspace(0,T,50)
    sigma_arr = sigma(x_arr)**2
    sigma_int = np.trapz(sigma_arr, x_arr)
    return np.sqrt(sigma_int/T)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    T = 2
    S0 = 1
    r = 0.045
    Klist = np.linspace(1e-7,3,6)
    data = []
    t = 0 # T-1e-7
    for K in Klist:
        data.append(V(S0, t, K))
        
    plt.plot(Klist, data)
    plt.xlabel("K")
    plt.ylabel("V")
    plt.show()