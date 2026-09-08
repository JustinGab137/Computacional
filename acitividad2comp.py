import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# System parameters
g = 9.81   # aceleración de la gravedad (m/s^2)
l = 1.0    # longitud del péndulo (m)
m = 1.0    # masa del péndulo (kg)
M = 4.0    # masa del bloque (kg)
k = 20.0   # constante elástica del resorte (N/m)

# Initial conditions
x0 = 0.3                     # desplazamiento inicial del bloque (m)
theta0 = np.radians(30.0)    # ángulo inicial del péndulo
v0, omega0 = 0.0, 0.0

# method parameters
tmax = 30
dt = 0.01
STRIDE = 2

# Dynamics of the moving-support pendulum (Eq. 1)
def dyn(t, y):
    x, v, theta, w = y
    st = sin(theta)
    ct = cos(theta)
    denom = M/m + st**2

    v_dot = ((g*ct + l*w**2) * st - (k/m)*x) / denom
    w_dot = -(1/l) * (g*(1 + M/m)*st + ct*(l*w**2*st - (k/m)*x)) / denom

    return np.array([v, v_dot, w, w_dot])

def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5*h, y + 0.5*k1)
    k3 = h * f(t + 0.5*h, y + 0.5*k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

n = int(tmax / dt)
t = np.linspace(0, n*dt, n+1)
y = np.empty((n+1, 4))
y[0] = np.array([x0, v0, theta0, omega0])

for i in range(n):
    y[i+1] = rk4(dyn, t[i], y[i], dt)

# separar variables
x = y[:, 0]
v = y[:, 1]
theta = y[:, 2]
omega = y[:, 3]

# Kinematics
xM, yM = x, np.zeros_like(x)                       # posición del bloque M
xm, ym = x + l*sin(theta), -l*cos(theta)           # posición de la masa m

# Figure and axis setup
fig, ax = plt.subplots(figsize=(9, 6))
R = l + np.max(np.abs(x)) + 0.5
ax.set(xlim=(-R, R + 1.5), ylim=(-l - 0.5, 0.5), aspect="equal",
       title="Péndulo con soporte móvil (resorte-masa-péndulo)")
ax.title.set_fontsize(14)
ax.tick_params(axis="both", labelsize=10)
ax.grid(alpha=0.3)

wall_x = -R - 0.3
ax.plot([wall_x, wall_x], [-0.4, 0.4], color="black", lw=3)  # pared

spring_line, = ax.plot([], [], "-", lw=1.5, color="gray")
block, = ax.plot([], [], "s", markersize=25, color="lightsteelblue",
                  markeredgecolor="black")
pend_line, = ax.plot([], [], "o-", lw=1.5, color="black", markersize=6)
bob, = ax.plot([], [], "o", markersize=14, color="lightcoral",
               markeredgecolor="black")
trace, = ax.plot([], [], "-", lw=1, color="red", alpha=0.5)
clock = ax.text(0.05, 0.93, "", transform=ax.transAxes, fontsize=11)

def spring_coords(x_end, n_coils=15, amp=0.08):
    xs = np.linspace(wall_x, x_end, n_coils*2)
    ys = amp * np.array([(-1)**i for i in range(len(xs))])
    ys[0] = 0
    ys[-1] = 0
    return xs, ys

def animate(i):
    xs, ys = spring_coords(xM[i])
    spring_line.set_data(xs, ys)
    block.set_data([xM[i]], [yM[i]])
    pend_line.set_data([xM[i], xm[i]], [yM[i], ym[i]])
    bob.set_data([xm[i]], [ym[i]])
    trace.set_data(xm[:i], ym[:i])
    clock.set_text(f"t = {i*dt:.1f} s")
    return spring_line, block, pend_line, bob, trace, clock

ani = FuncAnimation(fig, animate, frames=range(0, n+1, STRIDE),
                     interval=STRIDE, blit=True)
plt.tight_layout()
plt.show()