"""Simulation des chaotischen Dreifachpendels. """

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Raumdimension dim, Anzahl der Teilchen N und
# Anzahl der Zwangsbedingungen R.
dim = 2
N = 2
R = 2

# Massen der Pendelkörper [kg].
m1 = 1.0
m2 = 1.0

# Länge der Pendelstangen [m].
l1 = 0.6
l2 = 0.3

# Simulationsdauer T und Zeitschrittweite dt [s].
T = 10
dt = 0.002

# Anfangsauslenkung [rad].
phi1 = math.radians(130.0)
phi2 = math.radians(0.0)

# Vektoren der Anfangspositionen [m].
r01 = l1 * np.array([math.sin(phi1), -math.cos(phi1)])
r02 = r01 + l2 * np.array([math.sin(phi2), -math.cos(phi2)])

# Array mit den Komponenten der Anfangspositionen [m].
r0 = np.concatenate((r01, r02))

# Array mit den Komponenten der Anfangsgeschwindigkeit [m/s].
v0 = np.zeros(N * dim)

# Array der Masse für jede Komponente [kg].
m = np.array([m1, m1, m2, m2])

# Betrag der Erdbeschleunigung [m/s²].
g = 9.81

# Parameter für die Baumgarte-Stabilisierung [1/s].
alpha = 10.0
beta = alpha


def h(r):
    """Zwangsbedingungen h_a(r) """
    r = r.reshape(N, dim)
    d1 = r[0]
    d2 = r[1] - r[0]
    return np.array([d1 @ d1 - l1 ** 2,
                     d2 @ d2 - l2 ** 2])


def grad_h(r):
    """Gradient der Zwangsbed.:    g[a, i] =  dh_a / dx_i  """
    r = r.reshape(N, dim)
    g = np.zeros((R, N, dim))

    # Erste Zwangsbedingung.
    g[0, 0] = 2 * r[0]

    # Zweite Zwangsbedingung.
    g[1, 0] = 2 * (r[0] - r[1])
    g[1, 1] = 2 * (r[1] - r[0])

    return g.reshape(R, N * dim)


def hesse_h(r):
    """Hesse-Matrix:    H[a, i, j] =  d²h_a / (dx_i dx_j)  """
    h = np.zeros((R, N, dim, N, dim))

    # Erstelle eine dim x dim - Einheitsmatrix.
    E = np.eye(dim)

    # Erste Zwangsbedingung.
    h[0, 0, :, 0, :] = 2 * E

    # Zweite Zwangsbedingung.
    h[1, 0, :, 0, :] = 2 * E
    h[1, 0, :, 1, :] = -2 * E
    h[1, 1, :, 0, :] = -2 * E
    h[1, 1, :, 1, :] = 2 * E

    return h.reshape(R, N * dim, N * dim)


def dgl(t, u):
    r, v = np.split(u, 2)

    # Lege die externe Kraft fest.
    F_ext = m * np.array([0, -g, 0, -g])

    # Stelle die Gleichungen für die lambdas auf.
    grad = grad_h(r)
    hesse = hesse_h(r)
    F = - v @ hesse @ v - grad @ (F_ext / m)
    F += - 2 * alpha * grad @ v - beta ** 2 * h(r)
    G = (grad / m) @ grad.T

    # Berechne die lambdas.
    lam = np.linalg.solve(G, F)

    # Berechne die Beschleunigung mithilfe der newtonschen
    # Bewegungsgleichung inkl. Zwangskräften.
    a = (F_ext + lam @ grad) / m

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-6,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Zerlege den Ors- und Geschwindigkeitsvektor in die
# entsprechenden Vektoren für die zwei Massen.
r1, r2 = np.split(r, 2)
v1, v2 = np.split(v, 2)

# Berechne die tatsächliche Pendellänge für jeden Zeitpunkt.
len1 = np.linalg.norm(r1, axis=0)
len2 = np.linalg.norm(r1-r2, axis=0)

# Berechne die Gesamtenergie für jeden Zeitpunkt.
E_pot = m1 * g * r1[1, :] + m2 * g * r2[1, :]
E_kin = 0
E_kin += 0.5 * m1 * np.sum(v1**2, axis=0)
E_kin += 0.5 * m2 * np.sum(v2**2, axis=0)
E = E_kin + E_pot

# Gib eine Tabelle der Minimal- und Maximalwerte aus.
print(f'      minimal        maximal')
print(f'  l1: {np.min(len1):.7f} m    {np.max(len1):.7f} m')
print(f'  l2: {np.min(len2):.7f} m    {np.max(len2):.7f} m')
print(f'   E: {np.min(E):.7f} J    {np.max(E):.7f} J')

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 0.4])
ax.set_aspect('equal')
ax.grid()

# Erzeuge je einen Punktplot, für die Position des Massen.
p1, = ax.plot([0], [0], 'bo', markersize=8, zorder=5)
p2, = ax.plot([0], [0], 'ro', markersize=8, zorder=5)

# Erzeuge einen Linienplot für die Stangen.
lines, = ax.plot([0, 0], [0, 0], 'k-', zorder=4)


def update(n):
    # Aktualisiere die Position des Pendelkörpers.
    p1.set_data(r1[:, n])
    p2.set_data(r2[:, n])

    # Aktualisiere die Position der Pendelstangen.
    p0 = np.array((0, 0))
    points = np.array([p0, r1[:, n], r2[:, n]])
    lines.set_data(points.T)

    return p1, p2, lines


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=10, blit=True)
plt.show()

# %% 
""" Create 3D plot of trajectory with time as 3rd dimension """

def plot_3D(r2):
    # number of time steps of simulation
    time_steps = r2.shape[-1]
    
    t = np.arange(0, time_steps)
    
    # evolution of position of 2nd mass in Cartesian coordinates
    x_evo = r2[0] 
    y_evo = r2[1]
    
    # plot x and y versus time
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(t, x_evo, y_evo)
    ax.set_zlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_xlabel("t [s]")
    pass

# %% 
""" Create function executing the simulation with random initial displacements
and returning the position, i.e. the x and y coordinates of the second mass 
for easier data acquisition """

import random

def get_pos():
    
    # Raumdimension dim, Anzahl der Teilchen N und
    # Anzahl der Zwangsbedingungen R.
    dim = 2
    N = 2
    R = 2
    
    # Massen der Pendelkörper [kg].
    m1 = 1.0
    m2 = 1.0
    
    # Länge der Pendelstangen [m].
    l1 = 0.6
    l2 = 0.3
    
    # Simulationsdauer T und Zeitschrittweite dt [s].
    T = 10
    dt = 0.002
    
    # # Anfangsauslenkung [rad].
    # phi1 = math.radians(130.0)
    # phi2 = math.radians(0.0)
        
    # initial displacements
    phi1 = random.randint(45, 145)
    phi2 = random.randint(0, 0)
        
    # Vektoren der Anfangspositionen [m].
    r01 = l1 * np.array([math.sin(phi1), -math.cos(phi1)])
    r02 = r01 + l2 * np.array([math.sin(phi2), -math.cos(phi2)])
    
    # Array mit den Komponenten der Anfangspositionen [m].
    r0 = np.concatenate((r01, r02))
    
    # Array mit den Komponenten der Anfangsgeschwindigkeit [m/s].
    v0 = np.zeros(N * dim)
    
    # Array der Masse für jede Komponente [kg].
    m = np.array([m1, m1, m2, m2])
    
    # Betrag der Erdbeschleunigung [m/s²].
    g = 9.81
    
    # Parameter für die Baumgarte-Stabilisierung [1/s].
    alpha = 10.0
    beta = alpha   
    
    # Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
    u0 = np.concatenate((r0, v0))
    
    # Löse die Bewegungsgleichung.
    result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-6,
                                       t_eval=np.arange(0, T, dt))
    t = result.t
    r, v = np.split(result.y, 2)
    
    # Zerlege den Ors- und Geschwindigkeitsvektor in die
    # entsprechenden Vektoren für die zwei Massen.
    r1, r2 = np.split(r, 2)
    v1, v2 = np.split(v, 2)

    return r2

# %% 
""" Create training data """

def create_data():
    # compute a random trajectory
    trajectory = get_pos()
    time_steps = trajectory.shape[-1]
    
    # array of current position
    x_now = trajectory[0][:-1]
    y_now = trajectory[1][:-1]
    
    # array of position one time step ahead
    x_next = trajectory[0][1:]
    y_next = trajectory[1][1:]

    # input data X and target data y
    X = np.zeros((time_steps-1, 2))
    y = np.zeros((time_steps-1, 2))
    X[:, 0] = x_now
    X[:, 1] = y_now
    y[:, 0] = x_next
    y[:, 1] = y_next
    
    return X, y

from sklearn.neural_network import MLPRegressor

X_train, y_train = create_data()

for i in range(100):
    X_tmp, y_tmp = create_data()
    X_train = np.append(X_train, X_tmp, axis=0)
    y_train = np.append(y_train, y_tmp, axis=0)

from sklearn.preprocessing import Normalizer

# scaler = Normalizer()

# X_train = scaler.fit_transform(X_train)


nn = MLPRegressor(hidden_layer_sizes=(10,10,10), max_iter=1000).fit(X_train, y_train)

X_test, y_test = create_data()

# X_test = scaler.transform(X_test)

print(nn.score(X_train, y_train))
print(nn.score(X_test, y_test))

y_pred = nn.predict(X_test)

r_test = np.array((y_test[:, 0], y_test[:, 1]))
r_pred = np.array((y_pred[:, 0], y_pred[:, 1]))

# %%

plot_3D(r_test)
plot_3D(r_pred)

# %% Feed output as new input

steps = 10

y_pred = nn.predict([X_test[0]])
y_pred_list = [y_pred]

for i in range(steps):
    y_pred = nn.predict(y_pred)
    y_pred_list.append(y_pred)

# %% Versuche Steve Bruntons Ergebnisse zu reproduzieren

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

## Simulate the Lorenz System

dt = 0.01
T = 8
t = np.arange(0,T+dt,dt)
beta = 8/3
sigma = 10
rho = 28


nn_input = np.zeros((100*(len(t)-1),3))
nn_output = np.zeros_like(nn_input)

fig,ax = plt.subplots(1,1,subplot_kw={'projection': '3d'})


def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))

x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                  for x0_j in x0])

for j in range(100):
    nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
    nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
    x, y, z = x_t[j,:,:].T
    ax.plot(x, y, z,linewidth=1)
    ax.scatter(x0[j,0],x0[j,1],x0[j,2],color='r')
             
ax.view_init(18, -113)
plt.show()

# %% 

X_train = nn_input[:-100, :]
y_train = nn_output[:-100, :]

X_test = nn_input[-100:, :]
y_test = nn_output[-100:, :]

model = MLPRegressor(hidden_layer_sizes=(10, 10, 10), solver='sgd', 
                     max_iter=1000, tol=1e-5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, y_pred))