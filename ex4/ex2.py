import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

S0 = 15
r = 0.02
sigma = 0.7
K = 12
T = 1
ds = 5

def Bm(N):
    random_increments = np.random.normal(0.0, 1.0, N)  # the epsilon values
    brownian_motion = np.cumsum(random_increments)  # calculate the brownian motion
    brownian_motion = np.insert(brownian_motion, 0, 0.0) # insert the initial condition
    return brownian_motion

def d1(S, t):
    return (np.log(S/K) + (r+sigma**2/2)*(T-t)) / (sigma * np.sqrt(T-t))

def d2(S, t):
    return (np.log(S/K) + (r-sigma**2/2)*(T-t)) / (sigma * np.sqrt(T-t))

def phi(x):
    return norm(loc=0, scale=1).cdf(x)

def V(S, t):
    return S*phi(d1(S, t)) - K*np.exp(-r*(T-t))*phi(d2(S, t))

t = np.linspace(0, T, 50)
S = np.linspace(S0-ds, S0+ds)

V_res = np.zeros((len(t)-1, len(S)))

for i,t_i in enumerate(t[:-1]):
    for j,s in enumerate(S):
        V_res[i,j] = V(s, t_i)

S_plot, t_plot = np.meshgrid(S, t)
V_plot = V(S_plot, t_plot)

Bmotion = Bm(len(t_plot)-1)
Bmotion = Bmotion + S0

ax = plt.axes(projection="3d")
ax.plot_surface(S_plot, t_plot, V_plot)
plt.plot(Bmotion, np.linspace(0,1,len(t)))

ax.view_init(30, 90)
ax.set_xlabel("S")
ax.set_ylabel("t")

plt.show()

data = []
for i in range(0,len(t)):
    data.append(V(Bmotion[i], t[i]))

plt.plot(t, data)
plt.xlabel("t")
plt.ylabel("S")
plt.show()
