"""ADMM как alternating projections на пересечение двух выпуклых множеств в 2D.

U = круг {||x - c1|| <= r1}, V = полуплоскость {a^T x <= b}.
ADMM (scaled): x = P_U(z - w); z = P_V(x + w); w += x - z.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
})

C_DGA = "#4C72B0"   # x (синий)
C_ALM = "#55A868"   # z (зелёный)
C_RES = "#C44E52"   # резидуал (красный)

rng = np.random.default_rng(7)

# --- множества ---
c1 = np.array([0.0, 0.0])
r1 = 1.0
a = np.array([1.0, 0.6])
a = a / np.linalg.norm(a)
b = -0.45  # полуплоскость a^T x <= b: умеренный «серп» пересечения


def proj_U(p):
    """Проекция на круг."""
    d = p - c1
    nd = np.linalg.norm(d)
    if nd <= r1:
        return p.copy()
    return c1 + d / nd * r1


def proj_V(p):
    """Проекция на полуплоскость a^T x <= b."""
    s = a @ p - b
    if s <= 0:
        return p.copy()
    return p - s * a


# --- ADMM (scaled) ---
K = 20
x = np.array([1.7, 1.4])   # старт вне обоих множеств
z = x.copy()
w = np.zeros(2)

xs, zs, res = [], [], []
for k in range(K):
    x = proj_U(z - w)
    z = proj_V(x + w)
    w = w + (x - z)
    xs.append(x.copy())
    zs.append(z.copy())
    res.append(np.linalg.norm(x - z))

xs = np.array(xs)
zs = np.array(zs)
res = np.array(res)
res = np.clip(res, 1e-16, None)

# ============ ФИГУРА ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

# --- панель 1: геометрия ---
# полуплоскость a^T x <= b (заливка через сетку)
gx = np.linspace(-1.8, 2.2, 400)
gy = np.linspace(-1.6, 2.0, 400)
GX, GY = np.meshgrid(gx, gy)
halfplane = (a[0] * GX + a[1] * GY <= b).astype(float)
ax1.contourf(GX, GY, halfplane, levels=[0.5, 1.5],
             colors=[C_ALM], alpha=0.16)

# круг U
circ = Circle(c1, r1, facecolor=C_DGA, alpha=0.16, edgecolor=C_DGA, lw=1.2)
ax1.add_patch(circ)

# граница полуплоскости
# a^T x = b -> точки на линии
t = np.linspace(-3, 3, 2)
perp = np.array([-a[1], a[0]])
line_pt = a * b
lp = line_pt[None, :] + t[:, None] * perp[None, :]
ax1.plot(lp[:, 0], lp[:, 1], color=C_ALM, lw=1.3, alpha=0.8)

# подписи множеств
ax1.text(-0.55, -0.78, r"$U=\{\|x-c\|\leq r\}$", color=C_DGA, fontsize=10)
ax1.text(-1.7, -1.25, r"$V=\{a^\top x\leq b\}$", color=C_ALM, fontsize=10)

# траектория x (зигзаг x->z)
ax1.plot(xs[:, 0], xs[:, 1], "o-", color=C_DGA, ms=4, lw=1.2,
         label=r"$x_k=P_U(z-w)$", zorder=5)
ax1.plot(zs[:, 0], zs[:, 1], "s-", color=C_ALM, ms=3.5, lw=1.0,
         label=r"$z_k=P_V(x+w)$", zorder=5, alpha=0.9)

# тонкие стрелки шагов x_k -> z_k (нарушение x=z)
for k in range(0, K, 1):
    ax1.annotate("", xy=zs[k], xytext=xs[k],
                 arrowprops=dict(arrowstyle="->", color="0.5",
                                 lw=0.6, alpha=0.6))

# старт (z_0 = x_0) и связь с первой проекцией на круг
ax1.plot([1.7, xs[0, 0]], [1.4, xs[0, 1]], "--", color="0.55", lw=0.8)
ax1.scatter([1.7], [1.4], marker="*", s=120, color="k", zorder=6)
ax1.annotate("старт", (1.7, 1.4), textcoords="offset points",
             xytext=(6, 4), fontsize=9)

ax1.set_aspect("equal")
ax1.set_xlim(-1.8, 2.2)
ax1.set_ylim(-1.6, 2.0)
ax1.set_xlabel(r"$x_1$")
ax1.set_ylabel(r"$x_2$")
ax1.set_title("ADMM = чередование двух лёгких проекций")
ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.2)

# --- панель 2: резидуал ---
ax2.semilogy(np.arange(1, K + 1), res, "o-", color=C_RES, ms=4, lw=1.4)
ax2.set_xlabel(r"итерация $k$")
ax2.set_ylabel(r"$\|x_k - z_k\|$ (нарушение $x=z$)")
ax2.set_title("Сходимость к самому пересечению")
ax2.grid(True, which="both", alpha=0.25)
ax2.set_xlim(0.5, K + 0.5)

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/exp_admm_intersection.pdf",
            bbox_inches="tight")
fig.savefig("/tmp/exp_admm_intersection.png", dpi=140, bbox_inches="tight")
print("residual first/last:", res[0], res[-1])
print("final x:", xs[-1], "in U:", np.linalg.norm(xs[-1]-c1) <= r1+1e-9,
      "in V:", a @ xs[-1] <= b + 1e-9)
