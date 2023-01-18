import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,101)
h = x[1] - x[0]
print(h)
f = lambda x :  1 + 2*np.exp(3*x)
fdd = lambda x : 18*np.exp(3*x)

fdd1 = []
# central difference:
for i in range(1,len(x)-1):
    res = 1/h**2*(f(x[i]+h) - 2*f(x[i]) + f(x[i]-h))
    fdd1.append(res)
    
fdd2 = []
# taylor:
for i in range(2,len(x)-2):
    res = (1/h**2) * (-1/12*f(x[i]+2*h) + 4/3*f(x[i]+h) - 5/2*f(x[i]) + 4/3*f(x[i]-h) - 1/12*f(x[i]-2*h))
    fdd2.append(res)


plt.title("derivative + approximations")
plt.plot(x[1:-1], fdd(x[1:-1]),label="analytical")
plt.plot(x[1:-1], fdd1, label="central dif")
plt.plot(x[2:-2], fdd2, label="taylor")
plt.legend()
plt.show()

plt.figure(1)
plt.title("differences")
diff1 = abs(fdd(x[1:-1]) - fdd1)
diff2 = abs(fdd(x[2:-2]) - fdd2)
plt.loglog(x[1:-1], diff1, label="central")
plt.loglog(x[2:-2], diff2, label="taylor")
plt.legend()
plt.show()