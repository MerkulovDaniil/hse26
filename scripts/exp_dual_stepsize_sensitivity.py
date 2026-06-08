import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.size"] = 13

rng = np.random.default_rng(1)

# ---- QP: min 1/2 x^T A x - b^T x  s.t. C x = d ----
n, m = 20, 8
mu, L = 1.0, 10.0

# A SPD with cond ~ L/mu, eigenvalues spread in [mu, L]
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
eigs = np.linspace(mu, L, n)
A = Q @ np.diag(eigs) @ Q.T
A = 0.5 * (A + A.T)
Ainv = np.linalg.inv(A)

b = rng.standard_normal(n)
C = rng.standard_normal((m, n))
d = rng.standard_normal(m)

# Dual hessian M = C A^{-1} C^T  (SPD, m x m)
M = C @ Ainv @ C.T
M = 0.5 * (M + M.T)
evM = np.linalg.eigvalsh(M)
lmin, lmax = evM[0], evM[-1]
alpha_star = 2.0 / (lmax + lmin)
alpha_stab = 2.0 / lmax  # strict stability boundary

# Optimal primal/dual from KKT:
# A x - b + C^T lambda = 0  ->  x = A^{-1}(b - C^T lambda)
# C x = d -> C A^{-1}(b - C^T lambda) = d -> M lambda = C A^{-1} b - d
rhs = C @ Ainv @ b - d
lam_star = np.linalg.solve(M, rhs)


def x_of_lambda(lam):
    return Ainv @ (b - C.T @ lam)


# Dual ascent: gradient of dual g(lambda) = C x(lambda) - d ; ascent on dual = minus?
# Dual function q(lambda) = min_x L = ... ; grad_lambda q = C x(lambda) - d.
# Update lambda_{k+1} = lambda_k + alpha (C x_k - d) is gradient ASCENT on concave dual.
# Iteration matrix: lambda - lam* maps by (I - alpha M).
K = 150
# Panel A: a few representative steps (too slow / optimal / unstable);
# the full sweep is kept for Panel B below.
factors = [0.1, 1.0, 1.9, 2.05]
factors_full = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0, 2.05]

cmap = cm.get_cmap("coolwarm")
colors = [cmap(0.05 + 0.9 * (f / 2.05)) for f in factors]

ALM_COLOR = "#55A868"

traj_err = {}
final_err = {}
for f in sorted(set(factors) | set(factors_full)):
    alpha = f * alpha_star
    lam = np.zeros(m)
    errs = [np.linalg.norm(lam - lam_star)]
    for _ in range(K):
        x = x_of_lambda(lam)
        g = C @ x - d
        lam = lam + alpha * g
        e = np.linalg.norm(lam - lam_star)
        errs.append(min(e, 1e3))  # clip top
    traj_err[f] = np.array(errs)
    final_err[f] = errs[-1]

# ---- ALM rho=10 on same problem ----
# Augmented Lagrangian: min_x 1/2 x^T A x - b^T x + lam^T(Cx-d) + rho/2||Cx-d||^2
# x-update closed form: (A + rho C^T C) x = b - C^T lam + rho C^T d
# lam-update: lam += rho (C x - d).  This is dual ascent with step rho on the
# rho-regularized dual -> converges for any rho>0 (here alpha=rho consistent by construction).
rho = 10.0
H = A + rho * C.T @ C
Hinv = np.linalg.inv(H)
lam_alm = np.zeros(m)
alm_err = [np.linalg.norm(lam_alm - lam_star)]
for _ in range(K):
    x = Hinv @ (b - C.T @ lam_alm + rho * C.T @ d)
    lam_alm = lam_alm + rho * (C @ x - d)
    e = np.linalg.norm(lam_alm - lam_star)
    alm_err.append(min(e, 1e3))
alm_err = np.array(alm_err)
alm_final = alm_err[-1]

clip_lo = 1e-16
ks = np.arange(K + 1)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))

# Panel A
for f, c in zip(factors, colors):
    y = np.clip(traj_err[f], clip_lo, 1e3)
    axA.semilogy(ks, y, color=c, lw=1.8,
                 label=r"$\alpha=%.2f\,\alpha^*$" % f)
axA.semilogy(ks, np.clip(alm_err, clip_lo, 1e3), color=ALM_COLOR,
             lw=2.2, ls="--", label=r"ALM ($\rho=10$)")
axA.set_xlabel(r"итерация $k$")
axA.set_ylabel(r"$\|\lambda_k - \lambda^*\|$")
axA.set_title(r"(A) Сходимость двойственного подъёма")
axA.set_ylim(clip_lo, 1e3)
axA.grid(True, which="both", ls=":", alpha=0.4)
axA.legend(fontsize=11, loc="upper right", framealpha=0.9, ncol=1)

# Panel B: final error vs alpha/alpha* (dense sweep, no per-point legend)
fac_arr = np.array(factors_full)
fin_arr = np.array([final_err[f] for f in factors_full])
axB.semilogy(fac_arr, np.clip(fin_arr, clip_lo, 1e3), "o-",
             color="#4C72B0", lw=1.8, ms=6, label=r"двойственный подъём")
# stability boundary alpha = 2/lmax(M)  in units of alpha*
stab_in_units = alpha_stab / alpha_star
axB.axvline(stab_in_units, color="0.35", ls=":", lw=1.6,
            label=r"граница устойчивости $\alpha=2/\lambda_{\max}(M)$")
axB.axhline(alm_final, color=ALM_COLOR, ls="--", lw=2.0,
            label=r"ALM ($\rho=10$)")
axB.set_xlabel(r"$\alpha/\alpha^*$")
axB.set_ylabel(r"финальная ошибка $\|\lambda_K - \lambda^*\|$")
axB.set_title(r"(B) Чувствительность к шагу")
axB.set_ylim(clip_lo, 1e3)
axB.grid(True, which="both", ls=":", alpha=0.4)
axB.legend(fontsize=10, loc="center left", framealpha=0.9)

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/exp_dual_stepsize_sensitivity.pdf",
            bbox_inches="tight")
fig.savefig("/tmp/exp_dual_stepsize_sensitivity.png", dpi=140,
            bbox_inches="tight")

print("alpha_star =", alpha_star)
print("stab boundary (units alpha*) =", stab_in_units)
print("lmin,lmax M =", lmin, lmax)
print("final errs:", {f: final_err[f] for f in factors})
print("ALM final:", alm_final)
