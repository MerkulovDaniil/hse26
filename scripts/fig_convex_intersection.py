"""Схема: пересечение двух выпуклых множеств для метода альтернирующих проекций.

Те же множества, что и в эксперименте exp_admm_intersection.py:
U = круг {||x - c|| <= r}, V = полуплоскость {a^T x <= b}.
Цель схемы — наглядно показать НЕпустое пересечение (серп) и две лёгкие
проекции, на которые ADMM расщепляет трудную проекцию на пересечение.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
})

C_U = "#4C72B0"   # круг (синий)
C_V = "#55A868"   # полуплоскость (зелёный)
C_INT = "#C44E52" # пересечение (красный)

# те же множества, что в эксперименте
c = np.array([0.0, 0.0])
r = 1.0
a = np.array([1.0, 0.6]); a = a / np.linalg.norm(a)
b = -0.45

fig, ax = plt.subplots(figsize=(6.4, 5.4))

# заливка полуплоскости
gx = np.linspace(-1.9, 2.0, 600)
gy = np.linspace(-1.7, 1.9, 600)
GX, GY = np.meshgrid(gx, gy)
in_V = (a[0] * GX + a[1] * GY <= b)
in_U = ((GX - c[0])**2 + (GY - c[1])**2 <= r**2)

ax.contourf(GX, GY, in_V.astype(float), levels=[0.5, 1.5], colors=[C_V], alpha=0.14)
# круг U
circ = Circle(c, r, facecolor=C_U, alpha=0.16, edgecolor=C_U, lw=1.6)
ax.add_patch(circ)
# пересечение (серп) — насыщенная заливка
ax.contourf(GX, GY, (in_U & in_V).astype(float), levels=[0.5, 1.5],
            colors=[C_INT], alpha=0.45)

# границы
t = np.linspace(-3, 3, 2)
perp = np.array([-a[1], a[0]])
lp = (a * b)[None, :] + t[:, None] * perp[None, :]
ax.plot(lp[:, 0], lp[:, 1], color=C_V, lw=1.8)
th = np.linspace(0, 2*np.pi, 300)
ax.plot(c[0] + r*np.cos(th), c[1] + r*np.sin(th), color=C_U, lw=1.8)

# подписи множеств
ax.text(0.42, 0.62, r"$U$", color=C_U, fontsize=20, fontweight="bold")
ax.text(-1.45, -1.15, r"$V$", color=C_V, fontsize=20, fontweight="bold")

# подпись пересечения со стрелкой
cx, cy = -0.55, -0.30  # точка внутри серпа
ax.annotate(r"$U\cap V$", xy=(cx, cy), xytext=(-1.65, 0.95),
            color=C_INT, fontsize=14,
            arrowprops=dict(arrowstyle="->", color=C_INT, lw=1.4))

# проекции точки извне: на U и на V по отдельности
p = np.array([1.55, 1.35])
# на круг
d = p - c; pU = c + d/np.linalg.norm(d)*r
# на полуплоскость
s = a @ p - b; pV = p - s*a
ax.scatter(*p, marker="*", s=160, color="k", zorder=6)
ax.annotate("точка извне", p, textcoords="offset points", xytext=(4, 6), fontsize=10)
for q, col, lab in [(pU, C_U, r"$P_U$"), (pV, C_V, r"$P_V$")]:
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=0.85))
    ax.scatter(*q, s=28, color=col, zorder=6)

ax.set_aspect("equal")
ax.set_xlim(-1.9, 2.0)
ax.set_ylim(-1.7, 1.9)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.grid(True, alpha=0.18)

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/convex_intersection.png", dpi=150, bbox_inches="tight")
fig.savefig("/tmp/convex_intersection.png", dpi=150, bbox_inches="tight")
print("intersection non-empty:", bool((in_U & in_V).any()))
