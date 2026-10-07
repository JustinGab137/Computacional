import numpy as np
from numpy import cos, pi, sin, log
import matplotlib.pyplot as plt

#system parameters

q = 2
omega = 2/3
T = 2*pi/omega

#condiciones iniciales
x0 = 1.25
v0 = 0
t = 0

#gamma values

gamma_min = 0.9
gamma_max = 1.8
dgamma = 0.0001
gamma_values = np.arange(gamma_min, gamma_max + dgamma, dgamma)
n_orbitas = len(gamma_values)

#parametros del metodo numerico

nTrans = 300
nLyapunov = 600
steps_per_T = 300
dt = T/steps_per_T

#estado inicial y vector de estado

x = np.full(n_orbitas, x0)
v = np.full(n_orbitas, v0)
xi = np.full(n_orbitas, 1/np.sqrt(2))
eta = np.full(n_orbitas, 1/np.sqrt(2))
y = np.concatenate([x, v, xi, eta])

#Dinamica del sistema + ecuaciones variacionales

def dyn(t, y):
    x = y[:n_orbitas]
    v = y[n_orbitas:2*n_orbitas]
    xi = y[2*n_orbitas:3*n_orbitas]
    eta = y[3*n_orbitas:]
    
    dx = v
    dv = -1/q * v - sin(x) + gamma_values*cos(omega*t)
    
    dxi = eta
    deta = -1/q * eta - cos(x)*xi
    
    return np.concatenate([dx, dv, dxi, deta])

# Runge Kutta 4th order method
def rk4(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

#integracion numerica

for period in range(nTrans):
    for step in range(steps_per_T):
        t += dt
        y = rk4(dyn, t, y, dt)
    xi = y[2*n_orbitas:3*n_orbitas]
    eta = y[3*n_orbitas:]
    norm = np.sqrt(xi**2 + eta**2)
    y[2*n_orbitas:3*n_orbitas] = xi/norm
    y[3*n_orbitas:] = eta/norm    

#maximum Lyapunov exponent calculation
sum_log_norm = np.zeros(n_orbitas)
for period in range(nLyapunov):
    for step in range(steps_per_T):
        t += dt
        y = rk4(dyn, t, y, dt)
    xi = y[2*n_orbitas:3*n_orbitas]
    eta = y[3*n_orbitas:]
    norm = np.sqrt(xi**2 + eta**2)
    sum_log_norm += np.log(norm)
    y[2*n_orbitas:3*n_orbitas] = xi/norm
    y[3*n_orbitas:] = eta/norm

#maximum Lyapunov exponent calculation per unit time
lambda_max = sum_log_norm/(nLyapunov*T)
imax = np.argmax(lambda_max)
gamma_at_max = gamma_values[imax]  
lambda_max_at_max = lambda_max[imax]
print(f"Maximum Lyapunov exponent: {lambda_max_at_max:.6f} at gamma = {gamma_at_max:.6f}")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(gamma_values, lambda_max, color='red', lw=0.1)
ax.axhline(0, color='black', lw=0.5, ls='--')
ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$\lambda_{max}$', fontsize=16)
ax.set_title('Maximum Lyapunov Exponent vs Gamma', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.95, 0.95, f'Max λ = {lambda_max_at_max:.6f} at γ = {gamma_at_max:.6f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=12)
plt.tight_layout()
plt.savefig("lyapunov_exponent.pdf")
plt.show()  
        
        
