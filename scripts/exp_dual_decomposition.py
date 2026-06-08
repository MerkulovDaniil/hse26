"""
Двойственная декомпозиция как координация цен (Vandenberghe).
B агентов, у каждого локальная цель f_i(x_i) = 1/2 a_i x_i^2 - b_i x_i,
общий ресурс  sum_i x_i = R.
Лагранжиан разделяется: при цене lambda агент решает локально в закрытой форме
    x_i(lambda) = (b_i - lambda) / a_i.
Координатор обновляет цену:  lambda <- lambda + alpha (sum_i x_i - R).
  - ресурс перегружен (sum x_i > R)  -> цена растёт;
  - ресурс недоиспользован           -> цена падает.
Левая панель  — динамика цены lambda_k -> lambda*.
Центр         — локальные решения агентов x_i^(k).
Правая панель — сходимость к глобальному оптимуму (зазор по цели).
Воспроизводимо: фиксированный seed.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12.5,
    'legend.fontsize': 8.5, 'font.family': 'serif', 'mathtext.fontset': 'cm',
})

rng = np.random.default_rng(3)
B, R = 5, 5.0
a = rng.uniform(0.5, 2.5, B)      # кривизна (чувствительность) агентов
b = rng.uniform(-1.0, 3.0, B)     # «предпочтения» агентов

# Глобальный оптимум: KKT -> x_i=(b_i-lambda*)/a_i, sum x_i = R
lam_star = (np.sum(b / a) - R) / np.sum(1.0 / a)
x_star = (b - lam_star) / a
def total_obj(x):
    return np.sum(0.5 * a * x**2 - b * x)
f_star = total_obj(x_star)

N = 25
# недорелаксированный шаг: цена подстраивается ПОСТЕПЕННО (видна динамика рынка).
# фактор сжатия |1 - alpha * sum(1/a)| = 0.6
alpha = 0.4 / np.sum(1.0 / a)
lam = 0.0                          # старт: цена занижена -> спрос превышает ресурс -> цена растёт
lam_hist, x_hist, gap_hist = [], [], []
for _ in range(N):
    x = (b - lam) / a                       # broadcast: цена -> локальные решения
    lam = lam + alpha * (np.sum(x) - R)      # gather: невязка ресурса -> цена
    lam_hist.append(lam); x_hist.append(x.copy())
    gap_hist.append(abs(total_obj(x) - f_star))
x_hist = np.array(x_hist)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0))

ax1.axhline(lam_star, ls='--', color='gray', lw=1, label=r'$\lambda^\star$')
ax1.plot(lam_hist, '-o', color='#DD8452', ms=4, lw=1.8)
ax1.set_xlabel('итерация $k$'); ax1.set_ylabel(r'цена $\lambda_k$')
ax1.set_title('Динамика цены ресурса'); ax1.legend(); ax1.grid(alpha=0.3)

cols = plt.cm.viridis(np.linspace(0.1, 0.9, B))
for i in range(B):
    ax2.plot(x_hist[:, i], color=cols[i], lw=1.5)
    ax2.axhline(x_star[i], color=cols[i], ls=':', lw=0.8, alpha=0.6)
ax2.set_xlabel('итерация $k$'); ax2.set_ylabel(r'$x_i^{(k)}=(b_i-\lambda_k)/a_i$')
ax2.set_title(f'Локальные решения {B} агентов'); ax2.grid(alpha=0.3)

ax3.semilogy(np.maximum(gap_hist, 1e-16), '-o', color='#4C72B0', ms=4, lw=1.8)
ax3.set_xlabel('итерация $k$'); ax3.set_ylabel(r'$f(x_k)-f^\star$')
ax3.set_title('Сходимость к глобальному оптимуму'); ax3.grid(alpha=0.3, which='both')

fig.suptitle(rf'Координация цен: $B={B}$ агентов делят ресурс $\sum_i x_i = {R:.0f}$', y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig('/root/hse26_repo/files/exp4_decomposition.pdf', bbox_inches='tight')
fig.savefig('/tmp/exp4_decomposition.png', bbox_inches='tight', dpi=140)
print(f'saved exp4_decomposition; lambda*={lam_star:.4f}  final gap={gap_hist[-1]:.2e}')
