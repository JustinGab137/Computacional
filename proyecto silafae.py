import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.close('all')



# ============================================================

# Constantes fisicas

# ============================================================

v = 246.0                     # VEV electrodebil (GeV)

mh = 125.0                    # Masa del higgs del SM (GeV)

mZ = 91.19                    # Masa del boson Z (GeV)

GammaSM_h = 4.07e-3           # Ancho total del higgs del SM (GeV)

lambda_1 = mh**2 / (2*v**2)   # Fijado por m_h (Ec. 129)

BR_MAX = 0.11                 # Cota ATLAS (95% C.L.)



# Opciones (ver notas): 

FACTOR_SS = 4.0               # Reproduce Gamma(h -> SS) de la ecuacion (125)

NORMALIZACION_S_CORRECTA = True    # Usa lambda_tilde_S = lambda_S/4, como en Ecs. (103)-(106)



# ============================================================

# Rangos del escaneo (coherentes con perturbatividad |lambda|<4pi, Ec. 114)

# ============================================================

N = 2_000_000

LAMBDA_MAX = 4*np.pi

MU2_MIN,  MU2_MAX  = -5.0e4, 5.0e4      # mu_2^2 [GeV^2]  (Cuadro I)

MUS2_MIN, MUS2_MAX = -5.0e4, 1.0e6      # mu_S^2 [GeV^2]: ampliado para llegar a mS ~ 1 TeV



rng = np.random.default_rng(42)

mu2sq    = rng.uniform(MU2_MIN, MU2_MAX, N)

lambda_2 = rng.uniform(0.01, LAMBDA_MAX, N)

lambda_3 = rng.uniform(-LAMBDA_MAX, LAMBDA_MAX, N)

lambda_4 = rng.uniform(-LAMBDA_MAX, LAMBDA_MAX, N)

lambda_5 = rng.uniform(-LAMBDA_MAX, LAMBDA_MAX, N)

muS2     = rng.uniform(MUS2_MIN, MUS2_MAX, N)

lambda_S = rng.uniform(0.01, LAMBDA_MAX, N)

lambda_6 = rng.uniform(-LAMBDA_MAX, LAMBDA_MAX, N)

lambda_7 = rng.uniform(-LAMBDA_MAX, LAMBDA_MAX, N)



# ============================================================

# Espectro

# ============================================================

mHpm2 = mu2sq + 0.5*lambda_3*v**2

mH02  = mu2sq + 0.5*(lambda_3 + lambda_4 + lambda_5)*v**2

mA02  = mu2sq + 0.5*(lambda_3 + lambda_4 - lambda_5)*v**2

mS2   = muS2 + lambda_6*v**2



# ============================================================

# Restricciones

# ============================================================

mask_pos = (mHpm2 > 0) & (mH02 > 0) & (mA02 > 0) & (mS2 > 0)



mH0  = np.sqrt(np.where(mask_pos, mH02, np.nan))

mA0  = np.sqrt(np.where(mask_pos, mA02, np.nan))

mHpm = np.sqrt(np.where(mask_pos, mHpm2, np.nan))

mS   = np.sqrt(np.where(mask_pos, mS2, np.nan))



mask_jer = (mH0 < mHpm) & (mH0 < mA0)

mask_LEP = (mH0 + mA0) > mZ

mask_pert = ((np.abs(lambda_3) < 4*np.pi) & (np.abs(lambda_4) < 4*np.pi) &

             (np.abs(lambda_5) < 4*np.pi) & (np.abs(lambda_6) < 4*np.pi) &

             (np.abs(lambda_7) < 4*np.pi) & (lambda_S < 4*np.pi))



# Estabilidad (Ecs. 100-108)

lamS = lambda_S/4.0 if NORMALIZACION_S_CORRECTA else lambda_S

lam3_peor = lambda_3 + lambda_4 - np.abs(lambda_5)

s12 = np.sqrt(lambda_1*lambda_2)

s1S = np.sqrt(lambda_1*lamS)

s2S = np.sqrt(lambda_2*lamS)

diag  = (lambda_2 > 0) & (lambda_S > 0)

pares = ((lambda_3 + 2*s12 >= 0) & (lam3_peor + 2*s12 >= 0) &

         (lambda_6 + 2*s1S >= 0) & (lambda_7 + 2*s2S >= 0))



def condicion_tres_campos(l3):

    lb3 = np.minimum(l3, 0.0)

    lb6 = np.minimum(lambda_6, 0.0)

    lb7 = np.minimum(lambda_7, 0.0)

    prod = (l3 + 2*s12) * (lambda_6 + 2*s1S) * (lambda_7 + 2*s2S)

    expr = (np.sqrt(lambda_1*lambda_2*lamS)

            + lb3*np.sqrt(lamS) + lb6*np.sqrt(lambda_2) + lb7*np.sqrt(lambda_1)

            + np.sqrt(2*np.clip(prod, 0.0, None)))

    return expr >= 0



mask_est = diag & pares & condicion_tres_campos(lambda_3) & condicion_tres_campos(lam3_peor)



mask_range = ((mH0 > 10) & (mH0 < 300) & (mA0 < 400) & (mHpm < 400) &

              (mS > 1) & (mS < 1000))



# Higgs invisible: h -> H0H0

lambda_L = lambda_3 + lambda_4 + lambda_5

abierto_H0 = (2*mH0 < mh)

fac_H0 = np.clip(np.where(abierto_H0, 1.0 - (2*mH0/mh)**2, 0.0), 0.0, None)

Gamma_hH0H0 = np.where(abierto_H0, (lambda_L*v)**2/(32*np.pi*mh)*np.sqrt(fac_H0), 0.0)

mask_higgs_H0 = Gamma_hH0H0/(GammaSM_h + Gamma_hH0H0) < BR_MAX



# Higgs invisible: h -> SS

abierto_S = (2*mS < mh)

fac_S = np.clip(np.where(abierto_S, 1.0 - (2*mS/mh)**2, 0.0), 0.0, None)

Gamma_hSS = np.where(abierto_S, FACTOR_SS*(lambda_6*v)**2/(32*np.pi*mh)*np.sqrt(fac_S), 0.0)

mask_higgs_S = Gamma_hSS/(GammaSM_h + Gamma_hSS) < BR_MAX



# BR total (suma de anchos si ambos canales estan abiertos)

Gamma_inv = Gamma_hH0H0 + Gamma_hSS

BR_inv_total = Gamma_inv/(GammaSM_h + Gamma_inv)

mask_higgs_tot = BR_inv_total < BR_MAX



# ============================================================

# Cutflow acumulado

# ============================================================

cortes = [

    ("masas positivas (incl. m_S^2)", mask_pos),

    ("jerarquia inerte",              mask_jer),

    ("LEP",                           mask_LEP),

    ("perturbatividad",               mask_pert),

    ("estabilidad (copositividad)",   mask_est),

    ("rangos de masas",               mask_range),

    ("BR(h->H0H0) < 0.11",            mask_higgs_H0),

    ("BR(h->SS)  < 0.11",             mask_higgs_S),

    ("BR invisible total < 0.11",     mask_higgs_tot),

]

acum = np.ones(N, dtype=bool)

print(f"{'corte':35s} {'sobreviven':>10s}")

for nombre, m in cortes:

    acum &= m

    print(f"{nombre:35s} {acum.sum():10d}")

full_mask = acum

print(f"\nSobreviven {full_mask.sum()} de {N} puntos ({100*full_mask.mean():.3f} %)")



# Puntos que cumplen los cortes previos y fallan la restriccion individual indicada.
# Las mascaras pueden solaparse; la supervivencia final usa BR_inv_total.
base = (mask_pos & mask_jer & mask_LEP & mask_pert & mask_est & mask_range)
excl_H0 = base & ~mask_higgs_H0
excl_SS = base & ~mask_higgs_S


survivors = pd.DataFrame({

    "mH0": mH0[full_mask], "mA0": mA0[full_mask], "mHpm": mHpm[full_mask],

    "mS": mS[full_mask], "mu2sq": mu2sq[full_mask], "lambda_2": lambda_2[full_mask],

    "lambda_3": lambda_3[full_mask], "lambda_4": lambda_4[full_mask],

    "lambda_5": lambda_5[full_mask], "muS2": muS2[full_mask],

    "lambda_S": lambda_S[full_mask], "lambda_6": lambda_6[full_mask],

    "lambda_7": lambda_7[full_mask], "BR_inv": BR_inv_total[full_mask],

}).reset_index(drop=True)

print(survivors.describe().T[["min", "max"]])



# ============================================================

# Benchmark (distancia euclidea en las 4 masas, Ec. 134)

# ============================================================

target = np.array([100.0, 200.0, 230.0, 250.0])   # mH0, mA0, mHpm, mS (GeV)

if len(survivors) == 0:

    raise SystemExit("Ningun punto sobrevive: aumenta N o relaja los rangos.")

d2 = ((survivors.mH0 - target[0])**2 + (survivors.mA0 - target[1])**2 +

      (survivors.mHpm - target[2])**2 + (survivors.mS - target[3])**2)

bench = survivors.loc[d2.idxmin()].copy()

print(f"\nBenchmark (distancia al objetivo = {np.sqrt(d2.min()):.1f} GeV):")

print(f"lambda1 (fijo por m_h): {lambda_1:.4f}")

print(bench.to_string())

print(f"\nm(H0)+m(A0) = {bench.mH0 + bench.mA0:.2f} GeV | m(H±)-m(H0) = {bench.mHpm - bench.mH0:.2f} GeV")

print(f"2m(H0) = {2*bench.mH0:.2f} | 2m(S) = {2*bench.mS:.2f} | m_h = {mh}")



# ============================================================

# Grafica 1: plano (mH0, mA0)

# ============================================================

fig, ax = plt.subplots(figsize=(7.5, 5.8))

ax.set_xlim(0, 300); ax.set_ylim(0, 400)



# Regiones excluidas

xx = np.linspace(0, mZ, 50)

ax.fill_between(xx, 0, mZ - xx, color="red", alpha=0.25, zorder=0)

ax.plot(xx, mZ - xx, color="darkred", lw=1.2, ls="--")

ax.text(30, 25, "excluido por LEP\n" + r"($Z**\to**o H^0A^0$)", color="darkred",

        fontsize=8, ha="center", va="center")

dd = np.linspace(0, 300, 10)

ax.fill_between(dd, 0, dd, color="gray", alpha=0.18, zorder=0)

ax.plot(dd, dd, color="gray", lw=1.0, ls="--")

ax.text(235, 120, r"$m_{A^0}<m_{H^0}$" + "\n(H0 no es el más ligero)",

        color="dimgray", fontsize=8, ha="center", va="center")



# Puntos excluidos por h -> H0H0 y permitidos

ax.scatter(mH0[excl_H0], mA0[excl_H0], c="tab:red", s=3, alpha=0.35,

           label=r"excluidos por $h**\to**o H^0H^0$")

sc = ax.scatter(survivors.mH0, survivors.mA0, c=survivors.lambda_3,

                cmap="viridis", s=5, alpha=0.7, label="puntos permitidos")

ax.axvline(mh/2, color="steelblue", lw=1.0, ls=":")

ax.text(mh/2 + 3, 392, r"$m_{H^0}=m_h/2$", color="steelblue", fontsize=8, va="top",

        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))

ax.scatter([bench.mH0], [bench.mA0], color="black", marker="*", s=220,

           edgecolor="white", linewidth=0.8, zorder=5, label="benchmark elegido")

ax.set_xlabel(r"$m_{H^0}$ [GeV]  (candidato a materia oscura)")

ax.set_ylabel(r"$m_{A^0}$ [GeV]")

ax.set_title("Región de masas permitida en el IDSM: sector inerte")

ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

plt.colorbar(sc, ax=ax).set_label(r"$**\lambda**ambda_3$")

plt.tight_layout()

plt.savefig("idsm_plano_mH0_mA0.png", dpi=150)



# ============================================================

# Grafica 2: (mH0, mS) y (mS, lambda_6)

# ============================================================

fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.8))



# (a) mH0 vs mS

ax = axs[0]

ax.set_xlim(0, 300); ax.set_ylim(0, 1000)

ax.fill_between(dd, 0, dd, color="tab:blue", alpha=0.07, zorder=0)

ax.plot(dd, dd, color="gray", ls="--", lw=1.0)

ax.text(30, 940, r"$m_S > m_{H^0}$: $H^0$ más ligero", fontsize=8, color="dimgray")

ax.text(300, 15, r"$m_S < m_{H^0}$: $S$ más ligero", fontsize=8, color="dimgray", ha="right")

sc = ax.scatter(survivors.mH0, survivors.mS, c=survivors.lambda_7,

                cmap="coolwarm", s=5, alpha=0.7, label="puntos permitidos")

ax.axvline(mh/2, color="steelblue", lw=1.0, ls=":")

ax.axhline(mh/2, color="darkorange", lw=1.0, ls=":")

ax.text(mh/2 + 3, 560, r"$m_{H^0}=m_h/2$", color="steelblue", fontsize=8)

ax.text(295, mh/2 + 12, r"$m_S=m_h/2$", color="darkorange", fontsize=8, ha="right")

ax.scatter([bench.mH0], [bench.mS], color="black", marker="*", s=220,

           edgecolor="white", linewidth=0.8, zorder=5, label="benchmark elegido")

ax.set_xlabel(r"$m_{H^0}$ [GeV]"); ax.set_ylabel(r"$m_S$ [GeV]")

ax.set_title("Dos candidatos de materia oscura")

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)

plt.colorbar(sc, ax=ax).set_label(r"$**\lambda**ambda_7$")



# (b) mS vs lambda_6 con region excluida por h -> SS

ax = axs[1]

ms_line = np.linspace(1, mh/2 - 1e-3, 400)

l6_max = np.sqrt((BR_MAX/(1-BR_MAX))*GammaSM_h*32*np.pi*mh /

                 (FACTOR_SS*v**2*np.sqrt(1 - (2*ms_line/mh)**2)))

l6_max = np.minimum(l6_max, LAMBDA_MAX)

ax.fill_between(ms_line, l6_max, LAMBDA_MAX, color="red", alpha=0.18, zorder=0)

ax.fill_between(ms_line, -LAMBDA_MAX, -l6_max, color="red", alpha=0.18, zorder=0)

ax.text(mh/4, 0.5*LAMBDA_MAX, r"excluido por $h**\to**o SS$" + "\n" + r"(solo $|**\lambda**ambda_6|**\lambda**esssim 5**\to**imes10^{-3}$ sobrevive)",

        color="darkred", fontsize=8, ha="center")

ax.scatter(mS[excl_SS], lambda_6[excl_SS], c="tab:red", s=3, alpha=0.35,

           label=r"excluidos por $h**\to**o SS$")

sc = ax.scatter(survivors.mS, survivors.lambda_6, c=survivors.mH0,

                cmap="plasma", s=5, alpha=0.7, label="puntos permitidos")

ax.axvline(mh/2, color="darkorange", lw=1.0, ls=":")

ax.scatter([bench.mS], [bench.lambda_6], color="black", marker="*", s=220,

           edgecolor="white", linewidth=0.8, zorder=5, label="benchmark elegido")

ax.set_xlim(0, 1000); ax.set_ylim(-LAMBDA_MAX, LAMBDA_MAX)

ax.set_xlabel(r"$m_S$ [GeV]"); ax.set_ylabel(r"$**\lambda**ambda_6$")

ax.set_title(r"Portal de Higgs y cota de $h**\to**o SS$")

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)

plt.colorbar(sc, ax=ax).set_label(r"$m_{H^0}$ [GeV]")



plt.tight_layout()

plt.savefig("idsm_planos_mS.png", dpi=150)

plt.show()