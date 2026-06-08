"""Где ADMM реально выигрывает: fused LASSO / 1D-TV  min 1/2||x-y||^2 + lam||Dx||_1.
D — оператор первой разности (связывает соседние координаты), поэтому у prox-градиента
НЕТ замкнутого prox для lam||Dx||_1 (его нельзя применить напрямую). Прямой конкурент —
субградиентный метод, и он ползёт O(1/sqrt k). ADMM расщепляет z=Dx: x-шаг = разреженный
линейный солвер (один Холецкий трёхдиаг. I+rho D^T D), z-шаг = мягкий порог. На порядки
точнее за то же число итераций и за то же время.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
C_ADMM = "#C44E52"
C_SUB = "#4C72B0"

rng = np.random.default_rng(0)
n = 200
# истинный кусочно-постоянный сигнал
x_true = np.zeros(n)
for a, b, v in [(0, 40, 1.0), (40, 90, -0.5), (90, 130, 2.0), (130, 170, 0.3), (170, n, -1.2)]:
    x_true[a:b] = v
y = x_true + 0.3 * rng.standard_normal(n)
lam = 1.2

# оператор первой разности D: (n-1) x n
D = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], shape=(n - 1, n)).tocsr()
Dt = D.T.tocsr()


def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def obj(x):
    return 0.5 * np.sum((x - y) ** 2) + lam * np.sum(np.abs(D @ x))


# ---------- ADMM ----------
def admm(K, rho=5.0):
    M = (sp.identity(n) + rho * (Dt @ D)).tocsc()
    lu = spla.factorized(M)              # один раз
    x = y.copy(); z = D @ x; u = np.zeros(n - 1)
    hist = []
    for _ in range(K):
        x = lu(y + rho * (Dt @ (z - u)))
        Dx = D @ x
        z = soft(Dx + u, lam / rho)
        u = u + Dx - z
        hist.append(obj(x))
    return x, np.array(hist)


# ---------- субградиентный метод ----------
def subgradient(K, a0=0.04):
    x = y.copy()
    best = np.inf; hist = []
    for k in range(1, K + 1):
        g = (x - y) + lam * (Dt @ np.sign(D @ x))
        x = x - a0 / np.sqrt(k) * g
        best = min(best, obj(x))
        hist.append(best)                # лучшее значение (субградиент немонотонен)
    return x, np.array(hist)


x_ref, _ = admm(5000)                    # эталон f*
f_star = obj(x_ref)

K = 2000
x_admm, h_admm = admm(K)
x_sub, h_sub = subgradient(K)
gap_admm = np.clip(h_admm - f_star, 1e-14, None)
gap_sub = np.clip(h_sub - f_star, 1e-14, None)

# ============ ФИГУРА: 2 панели ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ax1.plot(y, color="0.7", lw=0.9, label="зашумлённый $y$")
ax1.plot(x_true, color="0.2", lw=1.4, ls="--", label="истинный сигнал")
ax1.plot(x_admm, color=C_ADMM, lw=1.8, label="ADMM (fused LASSO)")
ax1.set_xlabel("координата"); ax1.set_ylabel("значение")
ax1.set_title("Кусочно-постоянное восстановление")
ax1.legend(loc="upper right", fontsize=9, framealpha=0.92)
ax1.grid(alpha=0.25)

it = np.arange(1, K + 1)
ax2.semilogy(it, gap_sub, color=C_SUB, lw=1.8, label=r"субградиент $O(1/\sqrt{k})$")
ax2.semilogy(it, gap_admm, color=C_ADMM, lw=1.8, label="ADMM")
ax2.set_xlabel(r"итерация $k$"); ax2.set_ylabel(r"$f(x_k)-f^\star$")
ax2.set_title("Зазор по функции: ADMM против субградиента")
ax2.legend(loc="upper right", fontsize=9, framealpha=0.92)
ax2.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/exp_admm_fused_lasso.pdf", bbox_inches="tight")
fig.savefig("/tmp/exp_admm_fused_lasso.png", dpi=140, bbox_inches="tight")
# для подписи: на каком k субградиент достигает того, что ADMM даёт за 50 итераций
admm50 = gap_admm[49]
k_sub_match = np.argmax(gap_sub <= admm50) + 1 if np.any(gap_sub <= admm50) else None
print("f*=%.6f  ADMM gap@50=%.2e  ADMM gap@2000=%.2e  sub gap@2000=%.2e" %
      (f_star, admm50, gap_admm[-1], gap_sub[-1]))
print("k_sub to match ADMM@50:", k_sub_match)
