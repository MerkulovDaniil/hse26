import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 12

rng = np.random.default_rng(0)

# Discrete double Legendre transform.
# y must span the slope range of f on [-3,3] (f'(3) ~ 25), otherwise the
# conjugate is truncated near the boundary and f** droops below f.
N, M = 600, 1200
x = np.linspace(-3, 3, N)
y = np.linspace(-26, 26, M)


def conjugate(fx, x, y):
    # f*(y) = max_x (y*x - f(x))
    # shape (M, N): y[:,None]*x[None,:] - fx[None,:]
    vals = y[:, None] * x[None, :] - fx[None, :]
    return vals.max(axis=1)


def biconjugate(fx, x, y):
    fstar = conjugate(fx, x, y)
    # f**(x) = max_y (x*y - f*(y))
    vals = x[:, None] * y[None, :] - fstar[None, :]
    return vals.max(axis=1)


# Panel 1: convex f
f1 = 0.5 * x**2 + 0.2 * x**4
f1bb = biconjugate(f1, x, y)
err1 = np.max(np.abs(f1bb - f1))

# Panel 2: non-convex double well
f2 = 0.25 * (x**2 - 1) ** 2
f2bb = biconjugate(f2, x, y)

C_F = "#4C72B0"
C_BB = "#C44E52"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- Panel 1 ---
ax = axes[0]
ax.plot(x, f1, color=C_F, lw=3.0, label=r"$f(x)$", alpha=0.85)
ax.plot(x, f1bb, color=C_BB, lw=1.6, ls="--", label=r"$f^{**}(x)$")
ax.set_title(r"Выпуклая $f=0.5x^2+0.2x^4$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f$")
ax.legend(loc="upper center", frameon=False)


def _scimath(v):
    """1.2e-04 -> '1.2\\cdot10^{-4}' для mathtext."""
    s = f"{v:.1e}"
    m, e = s.split("e")
    return rf"{m}\cdot10^{{{int(e)}}}"


ax.text(
    0.5,
    0.06,
    r"$\max|f^{**}-f|=%s$" % _scimath(err1),
    transform=ax.transAxes,
    va="bottom",
    ha="center",
    fontsize=12,
)
ax.set_xlim(-2, 2); ax.set_ylim(-0.4, 5.6)

# --- Panel 2 ---
ax = axes[1]
ax.fill_between(
    x,
    f2bb,
    f2,
    where=(f2 - f2bb) > 1e-4,
    color="#DD8452",
    alpha=0.45,
    hatch="///",
    edgecolor="#DD8452",
    linewidth=0.0,
    label=r"зазор $f-f^{**}$",
)
ax.plot(x, f2, color=C_F, lw=3.0, label=r"$f(x)$", alpha=0.85)
ax.plot(x, f2bb, color=C_BB, lw=2.0, label=r"$f^{**}(x)$ (вып. оболочка)")
ax.set_title(r"Невыпуклая $f=0.25(x^2-1)^2$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f$")
ax.legend(loc="upper center", frameon=False)
ax.annotate(
    r"плоское дно $f^{**}=0$ на $[-1,1]$",
    xy=(0.0, 0.0),
    xytext=(0.0, 0.95),
    ha="center",
    fontsize=12,
    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0),
)
ax.set_xlim(-1.85, 1.85); ax.set_ylim(-0.12, 1.5)

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/exp_conj_biconjugate.pdf", bbox_inches="tight")
fig.savefig("/tmp/exp_conj_biconjugate.png", dpi=140)
print("err1 =", err1)
print("flat bottom min f2bb on [-1,1]:", f2bb[(x >= -1) & (x <= 1)].min(),
      f2bb[(x >= -1) & (x <= 1)].max())
