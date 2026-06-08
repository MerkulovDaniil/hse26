"""
Практический пример: WATER-FILLING — распределение мощности по каналам связи.
Реальная задача (Wi-Fi/LTE/DSL, OFDM): передатчик делит бюджет мощности P между n
частотными каналами с разным уровнем шума sigma_i^2, максимизируя суммарную скорость:
    max_p  sum_i log(1 + p_i / sigma_i^2)   при   sum_i p_i <= P,  p_i >= 0.
KKT даёт замкнутую форму:
    p_i = max(0, 1/nu - sigma_i^2),
где nu >= 0 — ДВОЙСТВЕННАЯ переменная бюджета мощности. Величина 1/nu = «уровень воды».
Картинка буквально как наливание воды: каналы с низким шумом (дно ниже воды) получают
мощность p_i = вода над дном; каналы с высоким шумом (дно выше воды) получают 0.
Двойственная переменная nu — НЕ абстракция: это уровень, до которого «залита вода».
Воспроизводимо: фиксированный seed.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 15, 'axes.labelsize': 16, 'axes.titlesize': 16,
    'legend.fontsize': 14, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

rng = np.random.default_rng(11)
n = 12
sigma2 = np.sort(rng.uniform(0.3, 3.0, n))      # уровни шума каналов (дно «сосудов»)
P = 6.0                                          # бюджет мощности

# уровень воды w=1/nu подбираем бисекцией так, чтобы sum max(0, w - sigma2) = P
def used(w):
    return np.sum(np.maximum(0.0, w - sigma2))
lo, hi = 0.0, sigma2.max() + P
for _ in range(200):
    w = 0.5 * (lo + hi)
    if used(w) > P: hi = w
    else: lo = w
water = 0.5 * (lo + hi)
p = np.maximum(0.0, water - sigma2)
nu = 1.0 / water
active = p > 1e-9
print(f'water level 1/nu = {water:.4f}, nu = {nu:.4f}, активных каналов {active.sum()}/{n}, sum p = {p.sum():.4f}')

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.3),
                               gridspec_kw={'width_ratios': [1.4, 1]})

# слева: «наливание воды»
idx = np.arange(n)
axL.bar(idx, sigma2, color='#9b8060', edgecolor='k', linewidth=0.5, label=r'шум $\sigma_i^2$ (дно)')
axL.bar(idx, p, bottom=sigma2, color='#4FA3D1', edgecolor='k', linewidth=0.5,
        label=r'мощность $p_i$ (вода)')
axL.axhline(water, color='#1f6f8b', lw=2, ls='--',
            label=r'уровень воды $1/\nu$')
# отметить неактивные каналы (p_i=0, ограничение p_i>=0 активно)
for i in idx[~active]:
    axL.text(i, sigma2[i] + 0.05, '×', ha='center', va='bottom', color='#b00', fontsize=16)
axL.scatter([], [], marker='x', color='#b00', s=60, label=r'$p_i=0$ (над водой)')
axL.set_xlabel('канал $i$'); axL.set_ylabel('уровень мощности')
axL.set_xticks(idx); axL.legend(loc='upper left')
axL.set_ylim(0, (sigma2 + p).max() * 1.28)

# справа: зависимость суммарной мощности от уровня воды -> как nu балансирует бюджет
ws = np.linspace(0, sigma2.max() + P, 400)
axR.plot(ws, [used(x) for x in ws], color='#4FA3D1', lw=2)
axR.axhline(P, ls=':', color='gray', lw=1.2); axR.text(0.05, P, ' бюджет $P$', va='bottom', fontsize=13, color='gray')
axR.axvline(water, ls='--', color='#1f6f8b', lw=1.6)
axR.text(water, 0.2, r' $1/\nu^\star$', color='#1f6f8b', fontsize=13)
axR.set_xlabel(r'уровень воды $1/\nu$'); axR.set_ylabel(r'$\sum_i p_i$')
axR.set_title(r'$\nu$ балансирует бюджет', pad=8)
axR.grid(alpha=0.3)

fig.tight_layout()
fig.savefig('/root/hse26_repo/files/exp_waterfilling.pdf', bbox_inches='tight')
fig.savefig('/tmp/exp_waterfilling.png', bbox_inches='tight', dpi=140)
print('saved exp_waterfilling')
