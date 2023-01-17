import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('HistoricalPrices_NASDAQ.csv', sep=",;", engine="python")
# df is in reverse -> oldest: index tmax, newest: index 0

tmax = len(df)-1
close = np.log(df["Close"]/df["Close"][tmax])
mu = close[0]/tmax
print("mu:", mu)
c1 = np.array(close[:-1])
c2 = np.array(close[1:])
dx = c2-c1
sig2 = (1/tmax)*np.sum((dx-mu)**2)
print("sigma²:",sig2)
sig = np.sqrt(sig2)
print("sigma:",sig)


df_new = pd.read_csv('HistoricalPrices_NASDAQ_new.csv', sep=",", engine="python")

### ex 2
# sep12, until now: 65 days, 46 without weekends ?
tnew = 46

#mu = -0.001
#sig = 0.0004

def Bm(N):
    random_increments = np.random.normal(0.0, 1.0, N)  # the epsilon values
    brownian_motion = np.cumsum(random_increments)  # calculate the brownian motion
    brownian_motion = np.insert(brownian_motion, 0, 0.0) # insert the initial condition
    return brownian_motion

iter = 10
for k in range(0,iter):
    S = df["Close"][0]
    W = [0]*(tnew+1)
    Wit = 1
    for i in range(0,Wit):
        W += Bm(tnew)
    W /= Wit
    data = [S]
    for i in range(0,tnew):
        S = S*np.exp(mu+sig*(W[i+1]-W[i]))
        data.append(S)
    plt.plot(range(tmax,tmax+tnew+1),data,c="orange")


plt.plot(range(0,tmax+1)[::-1],df["Close"])
plt.plot(range(tmax,tmax+tnew),df_new["Close"],c="green")
plt.show()
