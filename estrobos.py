import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi


def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6


def mapa_estroboscopico(delta, k, beta, gamma, omega,
                        n_orbitas, x_rng, v_rng,
                        Trans=300, Nkeep=1000, steps_per_T=100, seed=0):
    
    T = 2*pi/omega
    dt = T/steps_per_T

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(*x_rng, n_orbitas)
    v0 = rng.uniform(*v_rng, n_orbitas)
    y = np.concatenate((x0, v0))

    def dyn(t, y):
        x = y[:n_orbitas]
        v = y[n_orbitas:]
        dv = -delta*v + k*x - beta*x**3 + gamma*cos(omega*t)
        return np.concatenate([v, dv])

    x_strobe = np.empty((Nkeep, n_orbitas))
    v_strobe = np.empty((Nkeep, n_orbitas))
    save_index = 0

    for step in range((Trans + Nkeep)*steps_per_T):
        y = rk4(dyn, step*dt, y, dt)
        if (step + 1) % steps_per_T == 0:
            if (step + 1)//steps_per_T > Trans:
                x_strobe[save_index] = y[:n_orbitas]
                v_strobe[save_index] = y[n_orbitas:]
                save_index += 1

    X = x_strobe.ravel()
    V = v_strobe.ravel()
    C = np.tile(np.arange(n_orbitas), Nkeep)   # color por órbita
    return X, V, C



Xa, Va, Ca = mapa_estroboscopico(delta=0.02, k=-1, beta=5, gamma=8, omega=0.5,
                                 n_orbitas=5000, x_rng=(1.1, 1.7), v_rng=(-2, 2),
                                 Trans=300, Nkeep=2000)
mask = (Xa > 1.1) & (Xa < 1.7) & (Va > -2.5) & (Va < 2.5)
Xa, Va, Ca = Xa[mask], Va[mask], Ca[mask]


Xb, Vb, Cb = mapa_estroboscopico(delta=0.15, k=+1, beta=1, gamma=0.3, omega=1,
                                 n_orbitas=500, x_rng=(-1.5, 1.5), v_rng=(-1, 1),
                                 Trans=300, Nkeep=1000)


# ---------------- Figura con subplots ----------------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(14, 5.5))

axa.scatter(Xa, Va, c=Ca, cmap='rainbow', s=1, linewidths=0, rasterized=True)
axa.set_xlim(1.1, 1.7)
axa.set_ylim(-2.5, 2.5)
axa.text(0.95, 0.95, r'$\delta = 0.02$', transform=axa.transAxes,
         ha='right', va='top', fontsize=14)

axb.scatter(Xb, Vb, c=Cb, cmap='rainbow', s=1, linewidths=0, rasterized=True)
axb.set_xlim(-1.5, 1.5)
axb.set_ylim(-0.7, 1.2)
axb.text(0.95, 0.95, r'$\delta = 0.15$', transform=axb.transAxes,
         ha='right', va='top', fontsize=14)

for ax, etiqueta in zip((axa, axb), ('(a)', '(b)')):
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$\dot{x}$', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_box_aspect(0.8)
    ax.text(0.02, 0.95, etiqueta, transform=ax.transAxes,
            ha='left', va='top', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig("poincare_ab.pdf", dpi=300)
plt.show()