"""
Gradient Flow vs Gradient Descent, strongly-convex exponential rate,
and accelerated (Nesterov / Su-Boyd-Candes) flow.

Outputs: files/exp_gradient_flow.pdf  (3 panels)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

BLUE, RED, GRAY, GREEN = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# Panel 1: Gradient Flow (continuous limit) vs Gradient Descent (steps)
# ----------------------------------------------------------------------
A2 = np.array([[3.0, 0.0], [0.0, 30.0]])      # ill-conditioned, kappa = 10
L2 = np.max(np.linalg.eigvalsh(A2))
grad2 = lambda x: A2 @ x
f2 = lambda x: 0.5 * x @ A2 @ x
x0 = np.array([9.0, 8.0])

# Gradient flow: fine forward Euler (approx. continuous trajectory)
dt = 1e-3
T = 2.5
nf = int(T / dt)
xf = np.empty((nf, 2)); xf[0] = x0
for k in range(1, nf):
    xf[k] = xf[k - 1] - dt * grad2(xf[k - 1])

# Gradient descent: discrete steps, alpha = 1/L
alpha = 1.0 / L2
nd = 14
xd = np.empty((nd, 2)); xd[0] = x0
for k in range(1, nd):
    xd[k] = xd[k - 1] - alpha * grad2(xd[k - 1])

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
ax = axes[0]
gx = np.linspace(-10, 10, 300); gy = np.linspace(-10, 10, 300)
GX, GY = np.meshgrid(gx, gy)
GZ = 0.5 * (A2[0, 0] * GX**2 + A2[1, 1] * GY**2)
ax.contour(GX, GY, GZ, levels=np.linspace(5, 2000, 14), colors="0.82", linewidths=0.7)
ax.plot(xf[:, 0], xf[:, 1], color=BLUE, lw=2.4, label="Gradient flow ($\\alpha\\!\\to\\!0$)")
ax.plot(xd[:, 0], xd[:, 1], "o-", color=RED, ms=4.5, lw=1.2,
        label="Gradient descent ($\\alpha=1/L$)")
ax.plot(0, 0, "*", color="k", ms=13, zorder=5)
ax.set_title("Поток = непрерывный предел шагов")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
ax.set_aspect("equal")

# ----------------------------------------------------------------------
# Panel 2: strongly-convex exponential rate  f-f* <= e^{-2 mu t}(f0-f*)
# ----------------------------------------------------------------------
n = 60
eig = np.linspace(1.0, 40.0, n)        # mu = 1, L = 40
mu = eig.min()
xs0 = rng.standard_normal(n)
tt = np.linspace(0, 6, 400)
# f - f* = 1/2 sum eig_i x0_i^2 e^{-2 eig_i t}
gap = 0.5 * (eig * xs0**2)[None, :] * np.exp(-2 * eig[None, :] * tt[:, None])
gap = gap.sum(axis=1)
gap0 = gap[0]
ax = axes[1]
ax.semilogy(tt, gap, color=BLUE, lw=2.4, label="Gradient flow")
ax.semilogy(tt, gap0 * np.exp(-2 * mu * tt), "--", color=RED, lw=1.8,
            label="$e^{-2\\mu t}\\,(f_0-f^\\star)$")
ax.set_title("Сильно выпуклый: экспоненциальная скорость")
ax.set_xlabel("$t$"); ax.set_ylabel("$f(x(t))-f^\\star$")
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.set_ylim(1e-6, 2 * gap0)

# ----------------------------------------------------------------------
# Panel 3: accelerated (Nesterov) flow vs gradient flow
#   GF:  x' = -grad f
#   AGF: X'' + (3/t) X' + grad f(X) = 0
# ill-conditioned strongly-convex quadratic, small mu
# ----------------------------------------------------------------------
nA = 80
eigA = np.linspace(0.01, 0.10, nA)     # small, nearly flat curvature -> GF crawls,
xa0 = rng.standard_normal(nA)          #   momentum (1/t^2) clearly dominates
fstar = 0.0
fval = lambda X: 0.5 * np.sum(eigA * X**2)
gradA = lambda X: eigA * X

Tm = 50.0
dtm = 5e-4
t0 = 0.1                                # start away from the 3/t singularity
m = int((Tm - t0) / dtm)
ts = np.empty(m)

# Gradient flow
Xg = xa0.copy(); gf_gap = np.empty(m)
# Accelerated flow X'' + (3/t) X' + grad f = 0, state (X, V=X')
Xa = xa0.copy(); Va = np.zeros(nA); agf_gap = np.empty(m)
t = t0
for k in range(m):
    ts[k] = t
    gf_gap[k] = fval(Xg) - fstar
    agf_gap[k] = fval(Xa) - fstar
    # GF step
    Xg = Xg - dtm * gradA(Xg)
    # AGF step (semi-implicit Euler on the 2nd-order ODE)
    Va = Va - dtm * ((3.0 / t) * Va + gradA(Xa))
    Xa = Xa + dtm * Va
    t += dtm

ax = axes[2]
g0 = gf_gap[0]
ax.semilogy(ts, gf_gap, color=BLUE, lw=2.4, label="Gradient flow")
ax.semilogy(ts, np.maximum(agf_gap, 1e-12), color=GREEN, lw=2.0,
            label="Ускоренный поток")
# theory guides
ax.semilogy(ts[5:], g0 * ts[5] / ts[5:], "--", color=GRAY, lw=1.2, label="$\\mathcal{O}(1/t)$")
ax.semilogy(ts[5:], g0 * (ts[5] / ts[5:])**2, ":", color=RED, lw=1.4, label="$\\mathcal{O}(1/t^2)$")
ax.set_title("Ускорение: затухающее трение $3/t$")
ax.set_xlabel("$t$"); ax.set_ylabel("$f-f^\\star$")
ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=1)
ax.set_ylim(1e-7, 3 * g0)
ax.set_xlim(t0, Tm)

plt.tight_layout()
out = "files/exp_gradient_flow.pdf"
plt.savefig(out, bbox_inches="tight")
print("saved", out)
