# Oscilación forzada

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt


# system parameters

g = 9.81  # acceleration due to gravity
m = 1  # mass
k = -1  # spring constant
l = 1  # damping coefficient
F = 1  # forcing amplitude
om = 2/3*pi  # forcing frequency

# intial conditions

x0 = 1  # initial position
v0 = 0  # initial velocity

#method parameters

tmax = 300.0  # maximum time
dt = 0.001  # time step

# Dynamics: forced duffing oscillator

def dyn(t, y):
    x, v = y
    dx = v
    dv = -(k/m)*x - (l/m)*x**3 + (F/m)*cos(om*t)
    return np.array([dx, dv])

#fourth-order Runge-Kutta method

def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5*h, y + 0.5*k1)
    k3 = h * f(t + 0.5*h, y + 0.5*k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# integration using Runge-Kutta method

n = int(tmax / dt)  # number of time steps
t = np.linspace(0, n * dt, n+1)  # time array
y = np.empty((n+1, 2))  # array to store results
y[0] =([x0, v0])  # set initial conditions
for i in range(n):
    y[i+1] = rk4(dyn, t[i], y[i], dt)
    
# separate variables

x = y[:, 0]
v = y[:, 1]

# phase space

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, v, linewidth=0.2)
ax.set_xlabel('Position ', fontsize=22)
ax.set_ylabel('Velocity', fontsize=22)
ax.set_title('Phase Space')
ax.tick_params(axis='both', labelsize=20)
plt.tight_layout()
ax.set_box_aspect(0.65)
plt.savefig('phase_space2.pdf', dpi=300, bbox_inches='tight')
plt.show()    







