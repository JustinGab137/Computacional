import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, sin

# parametros del sistema

q, gamma, omega = 2, 1.1799, 2/3
T = 2*pi/omega

#para la eq:
#theta'' + 1/q theta' + sin(theta) = gamma*cos(omega*t)

# condiciones iniciales (varias orbitas, una por color)

n_orbitas = 500
rng = np.random.default_rng(0)
x0 = rng.uniform(-3, 3, n_orbitas)
v0 = rng.uniform(0, 2, n_orbitas)

# estado inicial
y = np.concatenate((x0, v0))

# parametros del metodo numerico

Trans = 0
Nkeep = 100
steps_per_T = 100
dt = T/steps_per_T

# Dinamica del sistema
def dyn(t, y):
    x = y[:n_orbitas]
    v = y[n_orbitas:]
    dx = v
    dv = -1/q * v - sin(x) + gamma*cos(omega*t)
    return np.concatenate([dx, dv])

# Runge Kutta 4th order method
def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# almacenamiento estroboscopico


x_strobe = np.empty((Nkeep, n_orbitas))
v_strobe = np.empty((Nkeep, n_orbitas))
save_index = 0

# integración

total_periodos = Trans + Nkeep
total_steps = total_periodos * steps_per_T
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi  # Función para envolver ángulos entre -pi y pi
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period > Trans:
            x_strobe[save_index] = wrap(y[:n_orbitas])
            v_strobe[save_index] = wrap(y[n_orbitas:])
            save_index += 1

# mapa estroboscopico

X = x_strobe.ravel()
V = v_strobe.ravel()
C = np.tile(np.arange(n_orbitas), Nkeep)   # color por orbita

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X, V, c=C, cmap='rainbow', s=1, linewidths=0, rasterized=True)
ax.set_xlabel(r'$\theta$', fontsize=16)
ax.set_ylabel(r'$\dot{\theta}$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(-3, 3.25)
ax.set_ylim(-0.5, 2.5)
ax.text(0.95, 0.95, r'$\gamma = 1.1799$', transform=ax.transAxes,
        ha='right', va='top', fontsize=14)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig("estroboscop.pdf")
plt.show()