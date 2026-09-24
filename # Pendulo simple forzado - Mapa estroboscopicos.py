#   Pendulo simple forzado - Mapa estroboscopico

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

# Definicion de parametros

m = 1.0
l = 1.0
g = 1.0

F0 = 0.01
om = 2/pi

T = 2*pi/om


# Condiciones iniciales


Ntheta = 35
Nv = 15

theta0 = np.linspace(-pi, pi, Ntheta)
v0 = np.linspace(-2.5, 2.5, Nv)

THETA0, V0 = np.meshgrid(theta0, v0)

theta = THETA0.ravel()
v = V0.ravel()

Ncond = len(theta)


# Parametros de integracion


Nperiodos = 2000
Npasos = 150

dt = T/Npasos

Trans = 100

# Dinamica del pendulo forzado

def dyn(t, y):

    theta = y[:, 0]
    v = y[:, 1]

    dtheta = v

    dv = -(g/l)*sin(theta) \
         + (F0/(m*l))*cos(om*t)

    return np.column_stack((dtheta, dv))


# Runge-Kutta de 4to orden

def rk4(f, t, y, h):

    k1 = h*f(t, y)

    k2 = h*f(t + h/2,y + k1/2)

    k3 = h*f(t + h/2, y + k2/2)

    k4 = h*f(t + h, y + k3)

    return y + (k1 + 2*k2 + 2*k3 + k4)/6


# Integracion numerica


y = np.column_stack((theta, v))

PE_theta = []
PE_v = []

t = 0.0


for periodo in range(Nperiodos):

    for paso in range(Npasos):

        y = rk4(
            dyn,
            t,
            y,
            dt
        )

        t += dt


    if periodo >= Trans:

        theta_mapa = (y[:, 0] + pi) % (2*pi) - pi

        PE_theta.append(theta_mapa.copy())
        PE_v.append(y[:, 1].copy())

PE_theta = np.array(PE_theta)
PE_v = np.array(PE_v)

# Grafica del mapa estroboscopico


fig, ax = plt.subplots(
    figsize=(10, 7)
)


colores = plt.cm.turbo(
    np.linspace(0, 1, Ncond)
)



for j in range(Ncond):

    ax.plot(
        PE_theta[:, j],
        PE_v[:, j],
        '.',
        markersize=0.45,
        color=colores[j],
        alpha=0.65,
        rasterized=True
    )


ax.set_xlim(-pi, pi)
ax.set_ylim(-2.6, 2.6)

ax.set_xlabel(
    r'$\theta$',
    fontsize=24
)

ax.set_ylabel(
    r'$\dot{\theta}$',
    fontsize=24
)

ax.tick_params(
    axis='both',
    labelsize=17
)


ax.set_xticks(
    [-pi, -pi/2, 0, pi/2, pi]
)

ax.set_xticklabels(
    [
        r'$-\pi$',
        r'$-\frac{\pi}{2}$',
        r'$0$',
        r'$\frac{\pi}{2}$',
        r'$\pi$'
    ]
)

ax.grid(
    True,
    alpha=0.18,
    linewidth=0.7
)

ax.spines['top'].set_linewidth(1.2)
ax.spines['right'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)

ax.set_title(
    'Mapa estroboscópico del péndulo forzado',
    fontsize=22,
    pad=15
)

ax.set_box_aspect(0.72)

plt.tight_layout()


plt.savefig(
    'mapa_estroboscopico_pendulo.pdf',
    bbox_inches='tight',
    dpi=400
)

plt.savefig(
    'mapa_estroboscopico_pendulo.png',
    bbox_inches='tight',
    dpi=400
)

plt.show()