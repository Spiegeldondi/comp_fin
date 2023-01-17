import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

T = 2

def Xt(n, f):
    sum = 0
    dt = T/n
    for i in range(0,n):
        sum += f((i*dt+(i+1)*dt)/2)*np.random.normal(0.0, dt)
    return sum

def runTest(f, N, m, Xfun, v=False):
    data = []
    for i in range(0,N):
        if v and i%(N/10) == 0: print(i)
        data.append(Xfun(m, f))
    return data

def run(N, m, v):
    f = lambda x : np.cos(x)
    data1 = runTest(f, N, m, Xt, v)
    print("cos - mean:", np.mean(data1), "var", np.var(data1))
    
    f = lambda x : np.exp(x)
    data2 = runTest(f, N, m, Xt, v)
    print("exp - mean:", np.mean(data2), "var", np.var(data2))
    
    plotFunc(data1, 0.8108)
    plotFunc(data2, (np.exp(1)**4)/2-0.5)
    
def plotFunc(data, var):
    plt.figure()
    plt.hist(data,bins=20,density=True)
    x = np.linspace(np.min(data),np.max(data),50)
    plt.plot(x, stats.norm.pdf(x, 0, var))
    plt.show() 
      
def plotExp():
    f = lambda x : np.exp(x)
    x = np.linspace(0,2,20)
    plt.plot(x, f(x))
    plt.show()

if __name__ == "__main__":
    """
    N: 100000 vs 10000
    m: 2 vs 50
    """
    N = 100000
    m = 2
    print("expected vars:", 0.8108, (np.exp(1)**4)/2-0.5)
    # runRndm(N, m, True)
    run(N, m, False)
