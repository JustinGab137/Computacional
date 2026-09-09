import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



# 1. LIBRERÍAS & PARÁMETROS

g = 9.81   # aceleración de la gravedad (m/s^2)
l = 1.0    # longitud del péndulo (m)
m = 1.0    # masa del péndulo (kg)
M = 4.0    # masa del bloque (kg)
k = 1   # constante elástica del resorte (N/m)

# Condiciones iniciales
x0 = 0               # desplazamiento inicial del bloque (m)
theta0 = np.radians(179)    # ángulo inicial del péndulo
v0, omega0 = 0.0, 0.0

# Parámetros del método numérico
tmax = 180
dt = 0.01
STRIDE = 12


# 2. DINÁMICA & MÉTODO

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

# 3. INTEGRACIÓN & CINEMÁTICA

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

# Cinemática: posición de cada cuerpo
xM, yM = x, np.zeros_like(x)              # posición del bloque M
xm, ym = x + l*sin(theta), -l*cos(theta)  # posición de la masa m


# 4. ANIMACIÓN


wall_x = min(xM.min(), 0) - 0.8

margin = 0.4
x_all = np.concatenate([xM, xm, [wall_x]])
y_all = np.concatenate([yM, ym])

xlim = (x_all.min() - margin, x_all.max() + margin)
ylim = (y_all.min() - margin, max(y_all.max(), 0) + margin)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set(xlim=xlim, ylim=ylim, aspect="equal",
       title="Péndulo con soporte móvil (resorte-masa-péndulo)")
ax.title.set_fontsize(14)
ax.title.set_color("white")
ax.set_xlabel("x (m)", fontsize=10, color="white")
ax.set_ylabel("y (m)", fontsize=10, color="white")
ax.tick_params(axis="both", labelsize=9, colors="white")
for spine in ax.spines.values():
    spine.set_color("white")
ax.grid(alpha=0.2, color="gray")

# Pared de referencia
ax.plot([wall_x, wall_x], [-0.4, 0.4], color="white", lw=3, zorder=3)

# Punto de anclaje del resorte (antes no existía)
ax.plot(wall_x, 0, marker="o", markersize=8, color="white", zorder=4)

# Suelo/línea de referencia horizontal para dar contexto espacial
ax.axhline(0, color="gray", lw=0.8, alpha=0.5, zorder=1)

spring_line, = ax.plot([], [], "-", lw=1.5, color="lightgray", zorder=2)
block, = ax.plot([], [], "s", markersize=25, color="gray",
                  markeredgecolor="white", label="Bloque M", zorder=5)
pend_line, = ax.plot([], [], "-", lw=1.5, color="white", zorder=4)
pivot, = ax.plot([], [], "o", markersize=5, color="white", zorder=5)
bob, = ax.plot([], [], "o", markersize=14, color="red",
               markeredgecolor="white", label="Masa m", zorder=6)
trace, = ax.plot([], [], "-", lw=1, color="red", alpha=0.4, zorder=2)
clock = ax.text(0.03, 0.95, "", transform=ax.transAxes, fontsize=11,
                 va="top", color="white",
                 bbox=dict(boxstyle="round", fc="black", ec="white", alpha=0.7))

legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
                    facecolor="black", edgecolor="white",
                    markerscale=0.7, labelspacing=1.2, handletextpad=0.8,
                    borderpad=0.8)
for text in legend.get_texts():
    text.set_color("white")

def spring_coords(x_end, n_coils=15, amp=0.08):
    """Genera las coordenadas de un resorte en zigzag entre la pared y el bloque."""
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
    pivot.set_data([xM[i]], [yM[i]])
    bob.set_data([xm[i]], [ym[i]])
    trace.set_data(xm[:i], ym[:i])
    clock.set_text(f"t = {i*dt:.1f} s")
    return spring_line, block, pend_line, pivot, bob, trace, clock

# El intervalo ahora refleja el tiempo real simulado por fotograma
# (antes interval=STRIDE hacía que la animación corriera casi instantánea
# sin relación con el dt real, luciendo "brusca")
ani = FuncAnimation(fig, animate, frames=range(0, n+1, STRIDE),
                     interval=dt*STRIDE*1000, blit=True)
plt.tight_layout()
plt.show()