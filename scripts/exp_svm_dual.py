"""
SVM как killer-пример: решаем именно ДВОЙСТВЕННУЮ задачу, а двойственные
переменные alpha_i имеют кристально ясный смысл — это ВЕСА ОПОРНЫХ ВЕКТОРОВ.
alpha_i > 0  тогда и только тогда, когда точка i лежит на отступе или нарушает его
(опорный вектор). Точки далеко от границы имеют alpha_i = 0 и не влияют на решение.

Прямая задача (soft-margin):  min 1/2||w||^2 + C sum xi_i  при y_i(w^T x_i + b) >= 1 - xi_i.
Двойственная задача:
    max_alpha  sum_i alpha_i - 1/2 sum_ij alpha_i alpha_j y_i y_j x_i^T x_j
    при        sum_i alpha_i y_i = 0,   0 <= alpha_i <= C.
Восстановление:  w = sum_i alpha_i y_i x_i.
Решаем двойственную QP напрямую (scipy SLSQP) — честно показываем, что оптимизируем дуал.
Воспроизводимо: фиксированный seed.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize

matplotlib.rcParams.update({
    'font.size': 14, 'axes.labelsize': 15, 'axes.titlesize': 15,
    'legend.fontsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

rng = np.random.default_rng(7)
n_per = 22
mu1, mu2 = np.array([-0.95, -0.55]), np.array([0.95, 0.6])
X1 = rng.standard_normal((n_per, 2)) * 0.82 + mu1
X2 = rng.standard_normal((n_per, 2)) * 0.82 + mu2
X = np.vstack([X1, X2])
y = np.concatenate([-np.ones(n_per), np.ones(n_per)])
N = len(y)
C = 1.0

# ── Двойственная QP ──
K = X @ X.T
Q = (y[:, None] * y[None, :]) * K          # Q_ij = y_i y_j x_i^T x_j
def neg_dual(a):       return 0.5 * a @ Q @ a - a.sum()
def neg_dual_grad(a):  return Q @ a - np.ones(N)
cons = ({'type': 'eq', 'fun': lambda a: a @ y, 'jac': lambda a: y},)
bnds = [(0.0, C)] * N
res = minimize(neg_dual, np.zeros(N), jac=neg_dual_grad, bounds=bnds,
               constraints=cons, method='SLSQP', options={'maxiter': 500, 'ftol': 1e-10})
alpha = np.clip(res.x, 0, C)

# ── Восстановление w, b ──
w = (alpha * y) @ X
sv = alpha > 1e-5                            # опорные векторы
on_margin = (alpha > 1e-5) & (alpha < C - 1e-5)
b = np.mean(y[on_margin] - X[on_margin] @ w) if on_margin.any() else \
    np.mean(y[sv] - X[sv] @ w)
print(f'#SV = {sv.sum()} из {N};  ||w||={np.linalg.norm(w):.3f};  dual_obj={-res.fun:.4f}')

# ── Графики ──
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                               gridspec_kw={'width_ratios': [1.25, 1]})

# слева: данные + граница + отступы, размер точки ∝ alpha
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.6, X[:, 0].max()+0.6, 300),
                     np.linspace(X[:, 1].min()-0.6, X[:, 1].max()+0.6, 300))
Z = (np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)
axL.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['#999', 'k', '#999'],
            linestyles=['--', '-', '--'], linewidths=[1, 1.8, 1])
axL.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=['#dce6f5', '#f5dce0'], alpha=0.5)
for cls, col, mk in [(-1, '#4C72B0', 'o'), (1, '#C44E52', 's')]:
    m = y == cls
    axL.scatter(X[m, 0], X[m, 1], c=col, marker=mk, s=28, edgecolor='k',
                linewidth=0.4, zorder=3)
# опорные векторы: обводим, размер ∝ alpha
axL.scatter(X[sv, 0], X[sv, 1], s=140 + 380 * alpha[sv] / C, facecolors='none',
            edgecolors='#33aa33', linewidths=2.2, zorder=4, label='опорные векторы')
axL.set_title(r'Двойственное решение $w=\sum_i \alpha_i y_i x_i$', pad=10)
axL.set_xlabel('$x_1$'); axL.set_ylabel('$x_2$'); axL.legend(loc='upper left')
axL.set_xlim(xx.min(), xx.max()); axL.set_ylim(yy.min(), yy.max())

# справа: alpha_i отсортированы — видно, что нетривиальны только опорные векторы
order = np.argsort(-alpha)
bar_cols = ['#33aa33' if s else '#bbbbbb' for s in sv[order]]
axR.bar(np.arange(N), alpha[order], color=bar_cols, edgecolor='k', linewidth=0.3)
axR.axhline(C, ls=':', color='gray', lw=1)
axR.text(N*0.7, C, ' $C$ (нарушители)', va='bottom', fontsize=13, color='gray')
axR.set_title(r'Значения $\alpha_i$', pad=10)
axR.set_xlabel('точки (по убыванию $\\alpha_i$)'); axR.set_ylabel(r'$\alpha_i$')
axR.text(0.97, 0.80, f'опорных: {sv.sum()} из {N}\nостальные $\\alpha_i=0$',
         transform=axR.transAxes, ha='right', fontsize=13,
         bbox=dict(boxstyle='round', fc='#eafaea', ec='#33aa33'))

fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig('/root/hse26_repo/files/exp_svm_dual.pdf', bbox_inches='tight')
fig.savefig('/tmp/exp_svm_dual.png', bbox_inches='tight', dpi=140)
print('saved exp_svm_dual')
