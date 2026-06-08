import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import brentq

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

# Course palette: warm (orange) <-> cool (blue), avoid yellow viridis.
cmap = cm.get_cmap("coolwarm")

# ---------------- Panel A: support lines ----------------
# f(x) = 0.5 x^2 + 0.3 x^4,  f'(x) = x + 1.2 x^3
def f(x):
    return 0.5 * x**2 + 0.3 * x**4

def fprime(x):
    return x + 1.2 * x**3

slopes = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])

def touch_x(y):
    lo, hi = -5.0, 5.0
    return brentq(lambda x: fprime(x) - y, lo, hi)

xs_touch = np.array([touch_x(y) for y in slopes])
fstar = slopes * xs_touch - f(xs_touch)

# ---------- Figure A: support lines (slide "Геометрия: зазор опорной прямой") ----------
figA, axA = plt.subplots(1, 1, figsize=(6.2, 5.0))
xx = np.linspace(-1.7, 1.7, 400)
axA.plot(xx, f(xx), color="black", lw=2.4, label=r"$f(x)=0.5\,x^2+0.3\,x^4$", zorder=5)

norm_s = (slopes - slopes.min()) / (slopes.max() - slopes.min())
for y, xt, fs, ns in zip(slopes, xs_touch, fstar, norm_s):
    col = cmap(ns)
    line = y * xx - fs  # l_y(x) = y x - f*(y)
    axA.plot(xx, line, color=col, lw=1.8, alpha=0.9)
    axA.plot(xt, f(xt), "o", color=col, ms=7, zorder=6)
    axA.plot(0, -fs, "s", color=col, ms=6, zorder=6, mfc="white", mew=1.6)

axA.axvline(0, color="gray", lw=0.8, ls=":")
axA.set_xlim(-1.7, 1.7)
axA.set_ylim(-3.2, 3.2)
axA.set_xlabel(r"$x$")
axA.set_ylabel(r"$y$")
axA.set_title(u"Опорные прямые: сдвиг $=-f^*(y)$", fontsize=14)
axA.legend(loc="upper center", framealpha=0.9)

sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(slopes.min(), slopes.max()))
sm.set_array([])
cb = figA.colorbar(sm, ax=axA, fraction=0.046, pad=0.02)
cb.set_label(u"наклон $y$")

i = 0
axA.annotate(r"$-f^*(y)$", xy=(0, -fstar[i]), xytext=(-0.95, -fstar[i] + 0.6),
             fontsize=15, color="black",
             arrowprops=dict(arrowstyle="->", color="black", lw=1.1))

figA.tight_layout()
figA.savefig("/root/hse26_repo/files/exp_conj_support_smoothness.pdf", bbox_inches="tight")
figA.savefig("/tmp/exp_conj_support_smoothness.png", dpi=140)

# ---------- Figure B: strong convexity <-> smoothness (slide "Сильная выпуклость f и гладкость f*") ----------
# Конкретный пример: f(x) = (mu/2) x^2  =>  f*(y) = y^2/(2 mu),  grad f*(y) = y/mu.
# Тогда grad f* — прямая с наклоном 1/mu, и липшицева константа = 1/mu видна буквально.
mus = np.array([0.5, 1.0, 3.0])
cols_B = ["#2166ac", "#6a3d9a", "#b2182b"]   # синий / фиолетовый / красный (без белого центра)

figB, axB = plt.subplots(1, 1, figsize=(6.2, 5.0))
yy = np.linspace(-6, 6, 300)
for mu, col in zip(mus, cols_B):
    g = yy / mu                      # grad f*(y) = y/mu
    axB.plot(yy, g, color=col, lw=2.6,
             label=fr"$\mu={mu:g}$,  наклон $1/\mu={1/mu:.2g}$")

axB.axhline(0, color="gray", lw=0.8, ls=":")
axB.axvline(0, color="gray", lw=0.8, ls=":")
axB.set_xlim(-6, 6)
axB.set_ylim(-3.4, 3.4)
axB.set_xlabel(r"$y$")
axB.set_ylabel(r"$\nabla f^*(y)=y/\mu$")
axB.set_title(u"Больше $\\mu$ $\\to$ положе $\\nabla f^*$", fontsize=15)
axB.legend(loc="upper left", framealpha=0.9)

# врезка с конкретной парой f / f* / grad f*
axB.text(0.97, 0.05,
         r"$f(x)=\dfrac{\mu}{2}x^2$" "\n"
         r"$f^*(y)=\dfrac{y^2}{2\mu}$" "\n"
         r"$\nabla f^*(y)=\dfrac{y}{\mu}$",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=15,
         linespacing=1.6,
         bbox=dict(boxstyle="round,pad=0.4", fc="#eef4fb", ec="#4C72B0", lw=1.2))

figB.tight_layout()
figB.savefig("/root/hse26_repo/files/exp_conj_smoothness_strong.pdf", bbox_inches="tight")
figB.savefig("/tmp/exp_conj_smoothness_strong.png", dpi=140)

print("touch_x:", xs_touch)
print("fstar:", fstar)
print("done")
