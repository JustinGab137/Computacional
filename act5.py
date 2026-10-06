import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt

# Parametros del sistema
delta = 0.1
alpha = 2.0
beta = 2.0
om = 1.2
T = 2*pi/om

# Condiciones iniciales
x0 = 1.0
xp0 = 1.0

# Valores de gamma
gamma_min = 0.1
gamma_max = 7.0
dgamma = 0.002
gamma_values = np.arange(gamma_min, gamma_max + dgamma, dgamma)
n_orbits = len(gamma_values)

# Parametros numericos
Trans = 200
Nkeep = 200
steps_per_T = 200
dt = T/steps_per_T

# Estado inicial
x = np.full(n_orbits, x0)
v = np.full(n_orbits, xp0)
y = np.concatenate((x, v))

# Dinamica
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -delta*v + alpha*x - beta*x**3 + gamma_values*cos(om*t)
    return np.concatenate([dx, dv])

# Runge-Kutta 4
def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# Almacenamiento de puntos estroboscopicos
x_strobe = np.empty((Nkeep, n_orbits))
v_strobe = np.empty((Nkeep, n_orbits))
save_index = 0

total_steps = (Trans + Nkeep) * steps_per_T
for step in range(total_steps):
    y = rk4(dyn, step*dt, y, dt)
    if (step + 1) % steps_per_T == 0:
        if (step + 1)//steps_per_T > Trans:
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# Diagramas de bifurcacion
def plot_bif(data, ylabel, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    G = np.tile(gamma_values, (Nkeep, 1))
    ax.scatter(G.ravel(), data.ravel(), s=0.6, color='red',
               linewidths=0, rasterized=True)
    ax.set_xlabel(r'$\gamma$', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(0, gamma_max)
    if ylim: ax.set_ylim(*ylim)
    ax.set_box_aspect(0.65)
    plt.tight_layout()
    plt.savefig(fname, format='pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)

plot_bif(x_strobe, r'$x$', 'Bifurcation_x.pdf')
plot_bif(v_strobe, r'$\dot{x}$', 'Bifurcation_v.pdf')