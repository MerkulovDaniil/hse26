import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'mathtext.fontset': 'cm',
})

x0 = 1.0
alpha = 0.3
beta2 = 0.5
n_iters = 35

def grad_L(x):
    return 2.0 * x

# GD
x_gd = np.zeros(n_iters + 1)
x_gd[0] = x0
for k in range(n_iters):
    x_gd[k + 1] = x_gd[k] - alpha * grad_L(x_gd[k])

# Simplified Adam (beta1=0, eps=0)
x_adam = np.zeros(n_iters + 1)
x_adam[0] = x0
v = grad_L(x0)**2
for k in range(n_iters):
    g = grad_L(x_adam[k])
    v = beta2 * v + (1 - beta2) * g**2
    x_adam[k + 1] = x_adam[k] - alpha * g / np.sqrt(v)

limit_x = alpha / 2
iters = np.arange(n_iters + 1)

fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))

ax.plot(iters, x_gd, color='black', linewidth=2.5,
        label='GD', alpha=0.85, zorder=3)
ax.plot(iters, x_adam, color='#e74c3c', linewidth=1.2,
        marker='o', markersize=3.5,
        label=rf'Adam ($\beta_1\!=\!0,\;\beta_2\!=\!{beta2}$)', alpha=0.85, zorder=4)

ax.axhline(y=limit_x, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.5)
ax.axhline(y=-limit_x, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.5)
ax.axhline(y=0, color='lightgray', linewidth=0.8)

ax.annotate(r'$+\alpha/2$', xy=(n_iters, limit_x),
            xytext=(n_iters + 1, limit_x + 0.035),
            fontsize=14, color='#e74c3c', va='center')
ax.annotate(r'$-\alpha/2$', xy=(n_iters, -limit_x),
            xytext=(n_iters + 1, -limit_x - 0.035),
            fontsize=14, color='#e74c3c', va='center')

ax.set_xlabel('Итерация')
ax.set_ylabel(r'$x_k$')
ax.legend(loc='upper right', framealpha=0.95, edgecolor='lightgray')
ax.grid(True, alpha=0.25)
ax.set_xlim(0, n_iters + 4)
ax.set_xticks(range(0, n_iters + 1, 5))

fig.tight_layout()

out = '/root/hse26_repo/files'
fig.savefig(f'{out}/exp_adam_limit_cycle.pdf', bbox_inches='tight', dpi=150)
fig.savefig(f'{out}/exp_adam_limit_cycle.png', bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved to {out}/exp_adam_limit_cycle.{{pdf,png}}")
