import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from random import random

N = 2_000_000
EXPONENTES = range(2, 9)    
P = np.array([0.5, 0])     


x = np.zeros(N)
y = np.zeros(N)
xn, yn = 0.5, 0.0

    
for i in range(N):
    r = random()
    xn = P[0]
    yn = P[1]
    if r < 0.1:
        P = np.array([0.05 * xn, 0.6 * yn])
    elif r < 0.20:
        P = np.array([0.05 * xn, - 0.5 * yn + 1.0])
    elif r < 0.40: 
        P = np.array([0.46 * xn - 0.15 * yn, 0.39 * xn + 0.38 * yn + 0.6])
    elif r < 0.60:
        P = np.array([0.47 * xn - 0.15 * yn, 0.17 * xn + 0.42 * yn + 1.1])
    elif r < 0.80:
        P = np.array([0.43 * xn + 0.28 * yn, -0.25 * xn + 0.45 * yn + 1.0])
    else:
        P = np.array([0.42 * xn + 0.26 * yn,- 0.35 * xn + 0.31 * yn + 0.7])
    x[i] = P[0]
    y[i] = P[1]    


def centrar_en_cuadrado(x, y):
    """Escala y centra los puntos dentro del cuadrado [0, 1] × [0, 1]."""
    escala = max(np.ptp(x), np.ptp(y))           # ptp = máximo - mínimo
    centro_x = (x.max() + x.min()) / 2
    centro_y = (y.max() + y.min()) / 2
    return (x - centro_x) / escala + 0.5, (y - centro_y) / escala + 0.5

def contar_cajas(x, y, n):
    """Marca las cajas de una malla n × n que contienen algún punto."""
    columna = np.minimum((x * n).astype(int), n - 1)
    fila = np.minimum((y * n).astype(int), n - 1)

    ocupadas = np.zeros((n, n), dtype=bool)
    ocupadas[fila, columna] = True
    return ocupadas


def graficar(x, y, k):
    """Dibuja la malla con ε = 1/2^k y devuelve (n, N(ε), Df)."""
    n = 2 ** k
    ocupadas = contar_cajas(x, y, n)
    n_cajas = ocupadas.sum()
    dimension = np.log(n_cajas) / np.log(n)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Cajas ocupadas (azul), malla y helecho
    azul = ListedColormap(["none", "#bcd9ea"])
    ax.imshow(ocupadas, origin="lower", extent=(0, 1, 0, 1),
              cmap=azul, interpolation="nearest")
    ax.set_xticks(np.linspace(0, 1, n + 1))
    ax.set_yticks(np.linspace(0, 1, n + 1))
    ax.grid(color="gray", linewidth=0.3)  
    ax.scatter(x, y, s=0.00001, color="green")

    # Estilo
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(rf"Dimensión fractal del helecho de Barnsley  "
                 rf"$\epsilon=\frac{{1}}{{2^{k}}}=\frac{{1}}{{{n}}}$")
    ax.text(0.02, 0.98,
            f"N = {N}\nn = {n},   N(ε) = {n_cajas}\nDf = {dimension:.6f}",
            transform=ax.transAxes, va="top", fontsize=8)

    plt.savefig(f"barnsley_eps_1_{n}.png", dpi=300, bbox_inches="tight")
    plt.show()
    return n, n_cajas, dimension


x, y = centrar_en_cuadrado(x, y)

#Para varios epsilon, graficar y calcular la dimensión fractal

resultados = [graficar(x, y, k) for k in EXPONENTES]

print(f"{'n':>5} {'N(ε)':>8} {'Df':>10}")
for n, n_cajas, dimension in resultados:
    print(f"{n:>5} {n_cajas:>8} {dimension:>10.6f}")
    
    
def contar_cajas_N(x, y, n):
    """N(ε): número de cajas ocupadas, sin construir la malla n×n."""
    col = np.minimum((x * n).astype(np.int64), n - 1)
    fil = np.minimum((y * n).astype(np.int64), n - 1)
    return np.unique(fil * n + col).size

K_TODOS  = np.arange(2, 17)      # ε = 1/2^k, k = 2..16
K_AJUSTE = np.arange(4, 10)      # intervalo de escalamiento (ajustable)

ln_inv_eps = K_TODOS * np.log(2)                       # ln(1/ε)
ln_N = np.log([contar_cajas_N(x, y, 2**k) for k in K_TODOS])

# Pendiente local: sirve para justificar el intervalo
pend_local = np.diff(ln_N) / np.diff(ln_inv_eps)
for k, s in zip(K_TODOS[1:], pend_local):
    print(f"k={k:2d}  ln(1/ε)={k*np.log(2):6.3f}  pendiente local={s:.3f}")

# Ajuste lineal solo en el intervalo de escalamiento
mask = np.isin(K_TODOS, K_AJUSTE)
Df, b = np.polyfit(ln_inv_eps[mask], ln_N[mask], 1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ln_inv_eps[mask], Df * ln_inv_eps[mask] + b,
        label=rf"$D_f = {Df:.4f}$")
ax.scatter(ln_inv_eps, ln_N, s=12)
ax.set_xlabel(r"$\ln(1/\epsilon)$")
ax.set_ylabel(r"$\ln N(\epsilon)$")
ax.legend(loc="upper left")
plt.savefig("regresion_barnsley.png", dpi=300, bbox_inches="tight")
plt.show()    
