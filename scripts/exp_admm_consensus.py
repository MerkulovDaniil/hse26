import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import cholesky, solve

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 13

# ---- palette ----
C_DUAL = "#4C72B0"
C_ALM = "#55A868"
C_ADMM = "#C44E52"
C_MARKET = "#DD8452"

rng = np.random.default_rng(42)

# ---- distributed least squares setup ----
N = 10        # machines
m = 30        # rows per machine
n = 15        # dim of x
x_true = rng.standard_normal(n)

A_list, b_list = [], []
for i in range(N):
    A = rng.standard_normal((m, n))
    b = A @ x_true + 0.05 * rng.standard_normal(m)
    A_list.append(A)
    b_list.append(b)

# centralized reference: stacked normal equations
AtA_sum = sum(A.T @ A for A in A_list)
Atb_sum = sum(A.T @ b for A, b in zip(A_list, b_list))
x_global = solve(AtA_sum, Atb_sum)

# ---- consensus ADMM ----
rho = 1.0
K = 120

# pre-factor Cholesky of (A_i^T A_i + rho I)
chol_factors = []
Atb_list = []
for A, b in zip(A_list, b_list):
    M = A.T @ A + rho * np.eye(n)
    L = cholesky(M)        # M = L L^T
    chol_factors.append(L)
    Atb_list.append(A.T @ b)

def chol_solve(L, rhs):
    # solve (L L^T) x = rhs
    y = solve(L, rhs)
    return solve(L.T, y)

X = np.zeros((N, n))       # x_i
U = np.zeros((N, n))       # u_i
z = np.zeros(n)

x1_traj = np.zeros((K + 1, N))   # first coord of each x_i
z1_traj = np.zeros(K + 1)
cons_res = np.zeros(K + 1)        # sum_i ||x_i - z||
glob_res = np.zeros(K + 1)        # ||z - x_global||

x1_traj[0] = X[:, 0]
z1_traj[0] = z[0]
cons_res[0] = np.sum(np.linalg.norm(X - z, axis=1))
glob_res[0] = np.linalg.norm(z - x_global)

for k in range(1, K + 1):
    # x-update (local)
    for i in range(N):
        rhs = Atb_list[i] + rho * (z - U[i])
        X[i] = chol_solve(chol_factors[i], rhs)
    # z-update (reduce: averaging)
    z = np.mean(X + U, axis=0)
    # u-update (local)
    U += X - z

    x1_traj[k] = X[:, 0]
    z1_traj[k] = z[0]
    cons_res[k] = np.sum(np.linalg.norm(X - z, axis=1))
    glob_res[k] = np.linalg.norm(z - x_global)

cons_res = np.clip(cons_res, 1e-16, None)
glob_res = np.clip(glob_res, 1e-16, None)

# ============ plotting ============
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
it = np.arange(K + 1)

# --- panel 1: consensus of first coordinate ---
ax = axes[0]
for i in range(N):
    ax.plot(it, x1_traj[:, i], color="0.6", lw=0.8, alpha=0.8,
            label="$x_i^{(1)}$" if i == 0 else None)
ax.plot(it, z1_traj, color=C_ADMM, lw=2.6, label="$z^{(1)}$ (консенсус)")
ax.axhline(x_global[0], color="0.2", ls="--", lw=1.0,
           label="централизованное")
ax.set_xlabel("итерация $k$")
ax.set_ylabel("первая координата")
ax.set_title("(1) Стягивание $x_i$ к консенсусу $z$")
ax.legend(fontsize=11, loc="best")
ax.grid(alpha=0.25)

# --- panel 2: residuals (log-y), start at k=1 (k=0 is the trivial z=0 init) ---
ax = axes[1]
it2 = it[1:]
ax.semilogy(it2, cons_res[1:], color=C_ADMM, lw=2.0,
            label=r"$\sum_i \|x_i - z\|$")
ax.semilogy(it2, glob_res[1:], color=C_DUAL, lw=2.0,
            label=r"$\|z - x_{\mathrm{centr}}\|$")
ax.set_xlabel("итерация $k$")
ax.set_ylabel("невязка")
ax.set_title("(2) Сходимость к консенсусу и оптимуму")
ax.legend(fontsize=11, loc="best")
ax.grid(alpha=0.25, which="both")

# --- panel 3: barplot z vs x_true vs centralized ---
ax = axes[2]
nshow = 8
idx = np.arange(nshow)
w = 0.27
ax.bar(idx - w, x_true[:nshow], width=w, color="0.5", label="$x_{\\mathrm{true}}$")
ax.bar(idx, x_global[:nshow], width=w, color=C_DUAL, label="централизованное")
ax.bar(idx + w, z[:nshow], width=w, color=C_ADMM, label="ADMM $z$")
ax.set_xlabel("координата")
ax.set_ylabel("значение")
ax.set_title("(3) Финальное $z$ = централизованное")
ax.set_xticks(idx)
ax.set_xticklabels([str(j + 1) for j in idx])
# headroom + горизонтальная легенда полосой сверху, чтобы не накрывать столбцы
_ymin, _ymax = ax.get_ylim()
ax.set_ylim(_ymin, _ymax + 0.34 * (_ymax - _ymin))
ax.legend(fontsize=9, loc="upper center", ncol=3, framealpha=0.92,
          edgecolor="0.8", columnspacing=1.0, handletextpad=0.4)
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/exp_admm_consensus.pdf", bbox_inches="tight")
fig.savefig("/tmp/exp_admm_consensus.png", dpi=140)

print("z vs x_global max diff:", np.max(np.abs(z - x_global)))
print("final consensus residual:", cons_res[-1])
print("final global residual:", glob_res[-1])
