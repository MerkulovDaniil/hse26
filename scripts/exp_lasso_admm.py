"""
LASSO через ADMM vs ISTA / FISTA.
Задача:  min_x  1/2 ||A x - b||^2 + lambda ||x||_1,   A: m x n, x* разрежен.
ADMM:    x = (A^T A + rho I)^{-1}(A^T b + rho (z - u));  z = soft(x+u, lambda/rho);  u += x - z.
ISTA:    x <- soft(x - t A^T(A x - b), lambda t),  t = 1/L,  L = sigma_max(A)^2.
FISTA:   ISTA + моментум Нестерова.
Панель — зазор по цели f(x_k) - f*  (лог-шкала) для ISTA / FISTA / ADMM.
Диагностика невязок ADMM вынесена на отдельный слайд лекции.
Честно: f* берётся как минимум по всем методам на большом числе итераций.
Воспроизводимо: фиксированный seed.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'font.family': 'serif', 'mathtext.fontset': 'cm',
})
C = {'ADMM': '#C44E52', 'ISTA': '#4C72B0', 'FISTA': '#55A868'}

# ── Данные ──
rng = np.random.default_rng(0)
m, n, k = 200, 500, 25
A = rng.standard_normal((m, n)) / np.sqrt(m)
x_true = np.zeros(n)
supp = rng.choice(n, k, replace=False)
x_true[supp] = rng.standard_normal(k)
b = A @ x_true + 0.01 * rng.standard_normal(m)
lam = 0.1 * np.max(np.abs(A.T @ b))

def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

def obj(x):
    return 0.5 * np.sum((A @ x - b) ** 2) + lam * np.sum(np.abs(x))

N = 400
AtA, Atb = A.T @ A, A.T @ b
L = np.linalg.norm(A, 2) ** 2          # липшиц градиента гладкой части

# ── ADMM ──
rho = 1.0
# Cholesky фактор (A^T A + rho I) — переиспользуем на всех итерациях
Lchol = np.linalg.cholesky(AtA + rho * np.eye(n))
def solve_chol(rhs):
    y = np.linalg.solve(Lchol, rhs)
    return np.linalg.solve(Lchol.T, y)
x = np.zeros(n); z = np.zeros(n); u = np.zeros(n)
admm_obj, r_pri, r_dual = [], [], []
for _ in range(N):
    x = solve_chol(Atb + rho * (z - u))
    z_old = z
    z = soft(x + u, lam / rho)
    u = u + x - z
    admm_obj.append(obj(x))
    r_pri.append(np.linalg.norm(x - z))            # ||x - z||
    r_dual.append(rho * np.linalg.norm(z - z_old)) # ||rho (z - z_old)||

# ── ISTA / FISTA ──
def ista(accel):
    x = np.zeros(n); y = x.copy(); t_mom = 1.0; hist = []
    for _ in range(N):
        x_new = soft(y - (1.0 / L) * (A.T @ (A @ y - b)), lam / L)
        if accel:
            t_new = (1 + np.sqrt(1 + 4 * t_mom ** 2)) / 2
            y = x_new + ((t_mom - 1) / t_new) * (x_new - x)
            t_mom = t_new
        else:
            y = x_new
        x = x_new
        hist.append(obj(x))
    return hist
ista_obj, fista_obj = ista(False), ista(True)

f_star = min(min(admm_obj), min(ista_obj), min(fista_obj))
gap = lambda h: np.maximum(np.array(h) - f_star, 1e-16)

# ── График: только сходимость по цели (диагностика невязок — на отдельном слайде) ──
fig, axL = plt.subplots(1, 1, figsize=(6.4, 4.4))
it = np.arange(1, N + 1)
axL.semilogy(it, gap(ista_obj), color=C['ISTA'], lw=2, label='ISTA')
axL.semilogy(it, gap(fista_obj), color=C['FISTA'], lw=2, label='FISTA')
axL.semilogy(it, gap(admm_obj), color=C['ADMM'], lw=2.4, label='ADMM')
axL.set_xlabel('итерация $k$'); axL.set_ylabel(r'$f(x_k) - f^\star$')
axL.set_title('Сходимость по функции'); axL.legend(); axL.grid(alpha=0.3, which='both')

fig.suptitle(rf'LASSO: $m={m},\ n={n},\ \|x^\star\|_0={k}$', y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig('/root/hse26_repo/files/exp3_lasso_admm.pdf', bbox_inches='tight')
fig.savefig('/tmp/exp3_lasso_admm.png', bbox_inches='tight', dpi=140)
print('saved exp3_lasso_admm; f*=%.6f  ADMM final gap=%.2e' % (f_star, gap(admm_obj)[-1]))
