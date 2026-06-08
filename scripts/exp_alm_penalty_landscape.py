import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 14

# Problem data.
# H -- objective Hessian: almost degenerate -> flat valley along the
#      eigenvector with the small eigenvalue (0.05), i.e. the x2 direction.
# A_c, b_c -- constraint matrix/rhs (called A, b in the lecture): A x = b.
H = np.diag([4.0, 0.05])
g_lin = np.array([0.0, 0.0])
A_c = np.array([[1.0, 1.0]])      # constraint x1 + x2 = 1, full row rank
b_c = np.array([1.0])

# KKT optimum (common for all rho): [H A^T; A 0][x;nu] = [-g; b]
KKT = np.block([[H, A_c.T], [A_c, np.zeros((1, 1))]])
rhs = np.concatenate([-g_lin, b_c])
sol = np.linalg.solve(KKT, rhs)
x_star = sol[:2]

# Grid
xs = np.linspace(-2, 3, 400)
ys = np.linspace(-2, 3, 400)
X, Y = np.meshgrid(xs, ys)

def f_val(X, Y):
    return 0.5 * (H[0, 0] * X**2 + H[1, 1] * Y**2) + (g_lin[0] * X + g_lin[1] * Y)

def penalty(X, Y, rho):
    r = A_c[0, 0] * X + A_c[0, 1] * Y - b_c[0]
    return f_val(X, Y) + 0.5 * rho * r**2

# Two panels: bare objective (rho=0) vs moderate penalty -- умеренный rho даёт
# выраженную, но хорошо обусловленную чашу (большой rho вырождает её в тонкий серп).
rhos = [0, 3]

vmin = 0.0
vmax = 8.0
levels = np.linspace(vmin, vmax, 17)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)

# constraint line x1 + x2 = 1 -> x2 = 1 - x1
line_x = np.array([-2, 3])
line_y = 1.0 - line_x

cf = None
for ax, rho in zip(axes, rhos):
    Z = penalty(X, Y, rho)
    Zc = np.clip(Z, vmin, vmax)
    cf = ax.contourf(X, Y, Zc, levels=levels, cmap='Blues', extend='max')
    ax.contour(X, Y, Zc, levels=levels, colors='white', linewidths=0.4, alpha=0.5)
    ax.plot(line_x, line_y, color='#C44E52', lw=2.6, label=r'$Ax=b$')
    ax.plot(x_star[0], x_star[1], 'o', color='black', ms=9,
            markeredgecolor='white', markeredgewidth=1.4, zorder=5)
    ax.set_title(r'$\rho = %g$' % rho, fontsize=15)
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_xlabel(r'$x_1$')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

axes[0].set_ylabel(r'$x_2$')

cbar = fig.colorbar(cf, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label(r'$\Phi_\rho(x)$', fontsize=16)

fig.savefig('/root/hse26_repo/files/exp_alm_penalty_landscape.pdf', bbox_inches='tight')
fig.savefig('/tmp/exp_alm_penalty_landscape.png', dpi=140, bbox_inches='tight')
print('x_star =', x_star)
