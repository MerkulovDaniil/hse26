"""
Dual ascent: теория vs эмпирика скорости сходимости.
Задача:  min_x  1/2 x^T A x - b^T x   при   C x = d.
Двойственный подъём:  x_k = A^{-1}(b - C^T u_k);  u_{k+1} = u_k + alpha (C x_k - d).
Двойственная функция вогнута с гессианом -M, M = C A^{-1} C^T.
Оптимальный шаг alpha = 2/(lmax+lmin) даёт линейную скорость rho = (kappa_d - 1)/(kappa_d + 1),
где kappa_d = cond(M) — обусловленность ДВОЙСТВЕННОЙ задачи.
Левая панель  — ||u_k - u*|| для набора задач с разной kappa_d.
Правая панель — эмпирический rho (наклон лог-графика) против теоретического.
Воспроизводимо: фиксированный seed.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 12, 'font.family': 'serif', 'mathtext.fontset': 'cm',
})

rng = np.random.default_rng(1)
n, m = 40, 12

def make_problem(kappa_d):
    """Строим A,C так, чтобы M = C A^{-1} C^T имела заданную обусловленность kappa_d."""
    # случайный ортонормированный базис для строк C
    C = rng.standard_normal((m, n))
    # выберем A = I (тогда M = C C^T); подправим спектр C, чтобы cond(C C^T)=kappa_d
    U, _, Vt = np.linalg.svd(C, full_matrices=False)
    sv = np.linspace(1.0, np.sqrt(kappa_d), m)        # сингулярные значения C
    C = U @ np.diag(sv) @ Vt
    A = np.eye(n)
    return A, C

kappas = [3, 10, 30, 100]
colors = ['#1f4e79', '#2c7fb8', '#E69F00', '#b2182b']  # палитра курса (без жёлтого viridis)
N = 80
emp_rho, theo_rho = [], []

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

for kap, col in zip(kappas, colors):
    A, C = make_problem(kap)
    Ainv = np.linalg.inv(A)
    M = C @ Ainv @ C.T
    ev = np.linalg.eigvalsh(M)
    lmin, lmax = ev[0], ev[-1]
    kappa_d = lmax / lmin
    b = rng.standard_normal(n)
    d = rng.standard_normal(m)
    # точное решение через KKT: [A  C^T; C  0][x; -u] = [b; d]
    KKT = np.block([[A, C.T], [C, np.zeros((m, m))]])
    rhs = np.concatenate([b, d])
    sol = np.linalg.solve(KKT, rhs)
    u_star = sol[n:]
    alpha = 2.0 / (lmax + lmin)               # оптимальный шаг
    u = np.zeros(m); err = []
    for _ in range(N):
        x = Ainv @ (b - C.T @ u)
        u = u + alpha * (C @ x - d)
        err.append(np.linalg.norm(u - u_star))
    err = np.array(err)
    axL.semilogy(err, color=col, lw=2, label=fr'$\kappa_d={kappa_d:.0f}$')
    # эмпирический rho — геом. среднее отношений на линейном участке
    seg = err[5:40]
    ratios = seg[1:] / seg[:-1]
    emp_rho.append(np.exp(np.mean(np.log(ratios))))
    theo_rho.append((kappa_d - 1) / (kappa_d + 1))

axL.set_xlabel('итерация $k$'); axL.set_ylabel(r'$\|u_k - u^\star\|$')
axL.set_title('Двойственный подъём: линейная сходимость'); axL.legend(); axL.grid(alpha=0.3, which='both')

# справа: коэффициент сжатия rho как функция kappa_d.
# rho = во сколько раз падает ошибка за одну итерацию (наклон лог-графика слева).
# Теория: rho = (kappa_d-1)/(kappa_d+1); точки — измеренный наклон. Чем больше kappa_d, тем rho->1.
kd_grid = np.logspace(np.log10(2.2), np.log10(140), 200)
axR.plot(kd_grid, (kd_grid - 1) / (kd_grid + 1), '-', color='k', lw=2.0,
         label=r'теория $\rho=\dfrac{\kappa_d-1}{\kappa_d+1}$', zorder=2)
kd_emp = [(1 + tr) / (1 - tr) for tr in theo_rho]   # восстановленные kappa_d
axR.scatter(kd_emp, emp_rho, s=110, c=colors, zorder=3, edgecolor='k', linewidth=0.7,
            label='эксперимент (наклон слева)')
axR.axhline(1.0, ls=':', color='gray', lw=1)
axR.text(2.6, 0.965, r'$\rho=1$: нет сходимости', fontsize=10, color='gray', va='top')
axR.set_xscale('log')
axR.set_xlabel(r'обусловленность двойственной задачи $\kappa_d$')
axR.set_ylabel(r'сжатие ошибки за шаг $\rho$')
axR.set_title('Чем хуже $\\kappa_d$, тем $\\rho$ ближе к 1 (медленнее)')
axR.set_ylim(0.3, 1.04); axR.legend(loc='lower right'); axR.grid(alpha=0.3, which='both')

fig.tight_layout()
fig.savefig('/root/hse26_repo/files/exp1_dual_rates.pdf', bbox_inches='tight')
fig.savefig('/tmp/exp1_dual_rates.png', bbox_inches='tight', dpi=140)
print('saved exp1_dual_rates')
for kap, er, tr in zip(kappas, emp_rho, theo_rho):
    print(f'  kappa~{kap:4d}: emp_rho={er:.4f}  theo_rho={tr:.4f}')
