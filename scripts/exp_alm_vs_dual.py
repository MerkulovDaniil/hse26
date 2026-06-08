"""ALM vs Dual ascent на одной QP-задаче, 2 режима: сильно выпуклый и вырожденный."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["font.size"] = 11

DGA = "#4C72B0"   # Dual ascent
ALM = "#55A868"   # ALM

n, m = 20, 8
rng = np.random.default_rng(0)

# Ортогональная случайная Q
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
eig_sc = np.linspace(1, 30, n)              # сильно выпуклый спектр
eig_dg = eig_sc.copy()
eig_dg[:5] = 0.0                            # вырожденный: первые 5 -> 0

A_sc = Q @ np.diag(eig_sc) @ Q.T
A_dg = Q @ np.diag(eig_dg) @ Q.T
A_sc = 0.5 * (A_sc + A_sc.T)
A_dg = 0.5 * (A_dg + A_dg.T)

b = rng.standard_normal(n)
C = rng.standard_normal((m, n))
x0 = rng.standard_normal(n)
d = C @ x0                                   # допустимость гарантирована

def f_val(A, x):
    return 0.5 * x @ A @ x - b @ x

def solve_kkt(A, b, C, d):
    """f* через KKT: [[A, C^T],[C,0]] [x;u] = [b; d]. lstsq для вырожденного случая."""
    K = np.block([[A, C.T], [C, np.zeros((m, m))]])
    rhs = np.concatenate([b, d])
    sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    x_star = sol[:n]
    return x_star

K_iter = 200
rho = 10.0

def run_dual(A, alpha):
    """Dual ascent: x = argmin_x L(x,u) при фикс u, затем u += alpha (Cx-d)."""
    u = np.zeros(m)
    A_pinv = np.linalg.pinv(A)            # для вырожденного A
    hist = []
    for _ in range(K_iter):
        # A x = b - C^T u  -> x = pinv(A) (b - C^T u)
        x = A_pinv @ (b - C.T @ u)
        hist.append(x.copy())
        u = u + alpha * (C @ x - d)
    return hist

def run_alm(A):
    u = np.zeros(m)
    M = A + rho * C.T @ C
    hist = []
    for _ in range(K_iter):
        x = np.linalg.solve(M, b - C.T @ u + rho * C.T @ d)
        hist.append(x.copy())
        u = u + rho * (C @ x - d)
    return hist

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

for ax, (A, eig, title) in zip(
    axes,
    [(A_sc, eig_sc, "Сильно выпуклая $f$"),
     (A_dg, eig_dg, r"Вырожденный гессиан $f$ (ker $\nabla^2 f \neq 0$)")],
):
    x_star = solve_kkt(A, b, C, d)
    f_star = f_val(A, x_star)

    # шаг для dual ascent: alpha = mu / sigma_max(C)^2, mu = наименьшее НЕнулевое собств. A
    mu = eig[eig > 1e-9].min()
    sig_max = np.linalg.svd(C, compute_uv=False).max()
    alpha = mu / sig_max**2

    hist_d = run_dual(A, alpha)
    hist_a = run_alm(A)

    err_d = np.clip([abs(f_val(A, x) - f_star) for x in hist_d], 1e-16, None)
    err_a = np.clip([abs(f_val(A, x) - f_star) for x in hist_a], 1e-16, None)

    it = np.arange(1, K_iter + 1)
    ax.semilogy(it, err_d, color=DGA, lw=2, label="Двойственный подъём")
    ax.semilogy(it, err_a, color=ALM, lw=2, label="ALM")
    ax.set_title(title)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$|f(x_k) - f^\star|$")
    ax.set_ylim(1e-16, max(err_d.max(), err_a.max()) * 3)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="center right", frameon=True, framealpha=0.92, edgecolor="0.8")

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/exp_alm_vs_dual.pdf", bbox_inches="tight")
fig.savefig("/tmp/exp_alm_vs_dual.png", dpi=140)
print("saved")
