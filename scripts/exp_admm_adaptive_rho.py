"""Диагностика ADMM: прямая/двойственная невязки + адаптивный выбор rho (Boyd et al. §3.4.1).

Задача: LASSO  min 1/2||Ax-b||^2 + lam||x||_1, расщепление x=z.
ADMM (scaled): x=(A^TA+rho I)^{-1}(A^Tb+rho(z-u)); z=soft(x+u, lam/rho); u+=x-z.
  прямая невязка r_k = x_k - z_k;  двойственная s_k = -rho (z_k - z_{k-1}).
Сравниваем фиксированные rho с адаптивным правилом балансировки невязок.
Честно: адаптивное rho ~= лучшему фиксированному, но БЕЗ ручной настройки;
плохо выбранное фиксированное rho кратно медленнее.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky, solve

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
C_R = "#C44E52"   # прямая невязка (красный)
C_S = "#4C72B0"   # двойственная невязка (синий)
C_AD = "#55A868"  # адаптивный (зелёный)

rng = np.random.default_rng(3)
m, n = 150, 300
A = rng.standard_normal((m, n)) / np.sqrt(m)
x_true = np.zeros(n); idx = rng.choice(n, 15, replace=False)
x_true[idx] = rng.standard_normal(15)
b = A @ x_true + 0.01 * rng.standard_normal(m)
lam = 0.1 * np.max(np.abs(A.T @ b))
AtA = A.T @ A; Atb = A.T @ b


def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def factor(rho):
    return cholesky(AtA + rho * np.eye(n))


def admm(rho0, adaptive, K=400, tol=1e-6):
    rho = rho0
    L = factor(rho)
    x = np.zeros(n); z = np.zeros(n); u = np.zeros(n)
    rs, ss = [], []
    refacts = 0
    kstop = K
    for k in range(K):
        rhs = Atb + rho * (z - u)
        x = solve(L.T, solve(L, rhs))
        z_old = z.copy()
        z = soft(x + u, lam / rho)
        u = u + x - z
        r = np.linalg.norm(x - z)                 # прямая
        s = np.linalg.norm(-rho * (z - z_old))    # двойственная
        rs.append(r); ss.append(s)
        if r < tol and s < tol:
            kstop = k + 1
            break
        if adaptive:  # правило балансировки невязок (Boyd §3.4.1)
            if r > 10 * s:
                rho *= 2; u /= 2; L = factor(rho); refacts += 1
            elif s > 10 * r:
                rho /= 2; u *= 2; L = factor(rho); refacts += 1
    return np.array(rs), np.array(ss), kstop, refacts


# --- прогон ---
fixed_rhos = [0.05, 0.5, 5.0]
fixed_runs = {ro: admm(ro, adaptive=False) for ro in fixed_rhos}
rs_ad, ss_ad, k_ad, refacts = admm(1.0, adaptive=True)

# ============ ФИГУРА: 2 панели ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# панель 1: невязки адаптивного прогона (обе -> 0)
it = np.arange(1, len(rs_ad) + 1)
ax1.semilogy(it, np.clip(rs_ad, 1e-12, None), color=C_R, lw=1.8,
             label=r"прямая $\|r_k\|=\|x_k-z_k\|$")
ax1.semilogy(it, np.clip(ss_ad, 1e-12, None), color=C_S, lw=1.8,
             label=r"двойственная $\|s_k\|$")
ax1.set_xlabel(r"итерация $k$"); ax1.set_ylabel("норма невязки")
ax1.set_title("Адаптивный $\\rho$: обе невязки $\\to 0$")
ax1.legend(loc="upper right", framealpha=0.92, edgecolor="0.8")
ax1.grid(True, which="both", alpha=0.25)

# панель 2: число итераций до tol — фиксированные rho vs адаптивный
labels = [f"$\\rho={ro:g}$" for ro in fixed_rhos] + ["адапт."]
iters = [fixed_runs[ro][2] for ro in fixed_rhos] + [k_ad]
colors = ["0.6", "0.6", "0.6", C_AD]
bars = ax2.bar(range(len(iters)), iters, color=colors, edgecolor="k", linewidth=0.6)
ax2.set_xticks(range(len(iters))); ax2.set_xticklabels(labels)
ax2.set_ylabel(r"итераций до $10^{-6}$")
ax2.set_title("Цена неудачного $\\rho$ против адаптивного")
ax2.grid(True, axis="y", alpha=0.25)
for rect, v in zip(bars, iters):
    ax2.annotate(str(v), (rect.get_x() + rect.get_width() / 2, v),
                 textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/exp_admm_adaptive_rho.pdf", bbox_inches="tight")
fig.savefig("/tmp/exp_admm_adaptive_rho.png", dpi=140, bbox_inches="tight")
print("fixed iters:", {ro: fixed_runs[ro][2] for ro in fixed_rhos})
print("adaptive iters:", k_ad, "refactorizations:", refacts)
