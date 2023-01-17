import numpy as np
import matplotlib.pyplot as plt

S = 1
# T = 5
m = 1
# dt = T/m
dt = 0.001
N = 10000
"""
ex1 task:
"""
kappa = 100/dt
theta = 1/(1e2*kappa)
# 2*kappa*theta>sigma**2
sigma = 0.5*np.sqrt(2)*np.sqrt(theta*kappa)

# kappa = 1.5/dt
# theta = 1/(1e5*kappa)
# sigma = 0.5*np.sqrt(2)*np.sqrt(theta*kappa)

print("kappa:", kappa)
print("theta:", theta)
print("sigma:", sigma)

def step(S, t, dWt):
    return S + a(t, S)*dt + b(t, S)*dWt

def a(t, S):
    return kappa*(theta-S)

def b(t, S):
    return sigma*np.sqrt(S)
    
def run(S, m):
    for i in range(m):
        S = step(S, i*dt, np.random.normal(0, dt))
        print(S)
    return S

if __name__ == "__main__":
    Srange = np.linspace(1e-7,10,20)
    data = []
    for S in Srange:
        c = 0
        ac = 0
        print(S)
        for i in range(N):
            ac += 1
            if(run(S, m) < 0): c+= 1   
        data.append(c/ac)


    print(data)
    plt.plot(Srange, data)
    plt.xlabel("S")
    plt.ylabel("V")
    plt.show()
 