import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.size"] = 14

# --- Strongly-convex QP, same as exp_alm_vs_dual: n=20, m=8, seed=0 ---
rng = np.random.default_rng(0)
n, m = 20, 8

# Build SPD Q (strongly convex objective): min 0.5 x'Qx + q'x  s.t. Ax = b
M = rng.standard_normal((n, n))
Q = M @ M.T + n * np.eye(n)      # well-conditioned SPD
q = rng.standard_normal(n)
A = rng.standard_normal((m, n))
x_feas = rng.standard_normal(n)
b = A @ x_feas

# --- Exact optimum via KKT (linear system) ---
KKT = np.block([[Q, A.T],
                [A, np.zeros((m, m))]])
rhs = np.concatenate([-q, b])
sol = np.linalg.solve(KKT, rhs)
x_star = sol[:n]
f_star = 0.5 * x_star @ Q @ x_star + q @ x_star

C = A  # constraint matrix; inner Hessian = Q + rho * A^T A


def alm_outer_iters(rho, tol=1e-8, cap=5000):
    """Augmented Lagrangian method. Inner min solved exactly (quadratic).
    Returns number of outer (dual) updates until |f-f*|<tol."""
    H = Q + rho * (A.T @ A)
    Hinv = np.linalg.inv(H)
    lam = np.zeros(m)
    for k in range(1, cap + 1):
        # minimize 0.5 x'Qx + q'x + lam'(Ax-b) + 0.5*rho*||Ax-b||^2
        g = q + A.T @ lam - rho * (A.T @ b)
        x = -Hinv @ g
        f = 0.5 * x @ Q @ x + q @ x
        if abs(f - f_star) < tol:
            return k
        lam = lam + rho * (A @ x - b)
    return cap


# Start from rho where the method already converges within budget (no cap plateau).
rhos = np.logspace(np.log10(0.3), 3, 25)
iters = np.array([alm_outer_iters(r) for r in rhos], dtype=float)
conds = np.array([np.linalg.cond(Q + r * (C.T @ C)) for r in rhos])

# Общая работа = (внешние итерации) x (стоимость внутреннего шага).
# Стоимость внутреннего шага растёт как sqrt(cond) (число итераций CG на внутренней
# квадратичной задаче). Оптимальный rho минимизирует это произведение.
total = iters * np.sqrt(conds)
sweet_idx = int(np.argmin(total))
sweet_rho = rhos[sweet_idx]

# --- Plot: 2 панели ---
DGA = "#4C72B0"
ALM = "#55A868"
RES = "#C44E52"

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11.2, 4.4))

# Панель 1: две встречные кривые (twin axis)
l1, = ax1.plot(rhos, iters, "o-", color=ALM, lw=2, ms=4.5,
               label=r"внешние итерации (падают)")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel(r"параметр штрафа $\rho$")
ax1.set_ylabel(r"внешние итерации ALM", color=ALM)
ax1.tick_params(axis="y", labelcolor=ALM)
ax1.grid(True, which="both", ls=":", alpha=0.4)
ax2 = ax1.twinx()
l2, = ax2.plot(rhos, conds, "s--", color=DGA, lw=2, ms=3.5,
               label=r"обусловленность $\kappa$ (растёт)")
ax2.set_yscale("log")
ax2.set_ylabel(r"$\kappa(Q+\rho A^{\!\top}\! A)$", color=DGA)
ax2.tick_params(axis="y", labelcolor=DGA)
ax1.set_title("Две встречные тенденции")
ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper center",
           framealpha=0.9, fontsize=9.5)

# Панель 2: общая работа = внешние x sqrt(cond), с явным минимумом
ax3.plot(rhos, total, "o-", color=RES, lw=2, ms=4.5)
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.axvline(sweet_rho, color="0.35", ls="-.", lw=1.5)
ax3.plot([sweet_rho], [total[sweet_idx]], "*", color="k", ms=16, zorder=6)
ax3.annotate(r"оптимум $\rho^\star\approx%.1f$" % sweet_rho,
             xy=(sweet_rho, total[sweet_idx]),
             xytext=(sweet_rho * 2.2, total[sweet_idx] * 3.0), fontsize=10.5,
             arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
ax3.set_xlabel(r"параметр штрафа $\rho$")
ax3.set_ylabel(r"суммарная работа: всего внутр. итераций")
ax3.set_title(r"Работа $\approx$ внешние $\times\,\sqrt{\kappa}$: минимум $=$ баланс")
ax3.grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle(r"Компромисс выбора $\rho$ в методе ALM", y=1.02, fontsize=15)
fig.tight_layout()
fig.savefig("/root/hse26_repo/files/exp_alm_rho_tradeoff.pdf", bbox_inches="tight")
fig.savefig("/tmp/exp_alm_rho_tradeoff.png", dpi=140, bbox_inches="tight")

print(f"f_star = {f_star:.6f}")
print(f"sweet rho = {sweet_rho:.3g}, iters={iters[sweet_idx]:.0f}, cond={conds[sweet_idx]:.3g}")
print(f"iters range: {iters.min():.0f}..{iters.max():.0f}")
print(f"cond range: {conds.min():.3g}..{conds.max():.3g}")
