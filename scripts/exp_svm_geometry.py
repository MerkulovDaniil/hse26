"""Геометрия SVM для вводного слайда: разделяющая гиперплоскость, зазор 2/||w||,
опорные векторы на отступе, один нарушитель со слабиной xi. Схематично и чисто.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13})

C_NEG = "#4C72B0"   # класс -1
C_POS = "#C44E52"   # класс +1
C_SV = "#33aa33"

# Нормаль w ∝ (1,1); граница w·x+b=0 ⇒ x+y=6; отступы x+y=4 и x+y=8.
# ||w|| подобрано так, что w·x+b=±1 на отступах: w=(1,1)/2, b=-3 ⇒ (x+y)/2-3=±1.
w = np.array([0.5, 0.5]); b = -3.0
def val(p): return w @ p + b

fig, ax = plt.subplots(figsize=(7.0, 5.4))

xs = np.linspace(-0.5, 8.5, 100)
ax.plot(xs, 6 - xs, color="k", lw=2.0, zorder=3)                 # граница
ax.plot(xs, 8 - xs, color="0.5", lw=1.3, ls="--", zorder=2)      # отступ +1
ax.plot(xs, 4 - xs, color="0.5", lw=1.3, ls="--", zorder=2)      # отступ -1
ax.fill_between(xs, 4 - xs, 8 - xs, color="0.85", alpha=0.4, zorder=0)

# Точки классов
neg = np.array([[0.6, 1.6], [1.6, 0.7], [1.1, 1.1], [0.4, 2.3], [2.2, 0.5]])
pos = np.array([[5.0, 4.6], [4.7, 5.2], [5.8, 4.4], [6.4, 4.9], [4.4, 6.0]])
# опорные векторы: лежат на отступах (x+y=4 и x+y=8)
sv_neg = np.array([[2.4, 1.6], [1.3, 2.7]])     # x+y=4
sv_pos = np.array([[4.6, 3.4], [3.2, 4.8]])     # x+y=8

ax.scatter(neg[:, 0], neg[:, 1], c=C_NEG, marker="o", s=55, edgecolor="k", lw=0.5, zorder=4)
ax.scatter(pos[:, 0], pos[:, 1], c=C_POS, marker="s", s=55, edgecolor="k", lw=0.5, zorder=4)
ax.scatter(sv_neg[:, 0], sv_neg[:, 1], c=C_NEG, marker="o", s=55, edgecolor="k", lw=0.5, zorder=4)
ax.scatter(sv_pos[:, 0], sv_pos[:, 1], c=C_POS, marker="s", s=55, edgecolor="k", lw=0.5, zorder=4)
for p in np.vstack([sv_neg, sv_pos]):
    ax.scatter(*p, s=240, facecolors="none", edgecolors=C_SV, lw=2.4, zorder=5)

# Стрелка зазора 2/||w|| вдоль нормали, через точку на границе
n = w / np.linalg.norm(w)
c0 = np.array([3.0, 3.0])                  # точка на границе (x+y=6)
half = 1.0 / np.linalg.norm(w)             # геом. полуотступ = 1/||w||
p_lo, p_hi = c0 - half * n, c0 + half * n
ax.add_patch(FancyArrowPatch(p_lo, p_hi, arrowstyle="<->", color="k",
                             lw=1.6, mutation_scale=14, zorder=6))
ax.text(c0[0] + 0.35, c0[1] + 0.35, r"зазор $=\dfrac{2}{\|w\|}$", fontsize=13)

# Стрелка нормали w
ax.add_patch(FancyArrowPatch([5.0, 1.0], [5.0, 1.0] + 1.3 * n, arrowstyle="-|>",
                             color="k", lw=1.6, mutation_scale=14, zorder=6))
ax.text(5.05, 1.0 + 1.4 * n[1], r"$w$", fontsize=14)

# Нарушитель: точка класса +1 за своим отступом (внутри полосы), слабина xi
viol = np.array([3.6, 2.6])                # x+y=6.2 < 8 ⇒ нарушает свой отступ
ax.scatter(*viol, c=C_POS, marker="s", s=55, edgecolor="k", lw=0.5, zorder=4)
foot = viol + ((8 - (viol[0] + viol[1])) / 2.0) * (2 * n)   # проекция на свой отступ x+y=8
ax.plot([viol[0], foot[0]], [viol[1], foot[1]], color=C_POS, lw=1.4, ls=":", zorder=4)
ax.text(viol[0] + 0.15, viol[1] - 0.35, r"$\xi_i$", fontsize=13, color=C_POS)

ax.text(0.5, 3.0, r"класс $-1$", color=C_NEG, fontsize=12, rotation=0)
ax.text(5.4, 5.6, r"класс $+1$", color=C_POS, fontsize=12)
ax.text(5.9, 0.05, r"опорные векторы", color=C_SV, fontsize=11)

ax.set_xlim(-0.3, 8.3); ax.set_ylim(-0.3, 6.6)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
ax.set_aspect("equal")
ax.set_title("Максимальный зазор и опорные векторы")

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/exp_svm_geometry.pdf", bbox_inches="tight")
fig.savefig("/tmp/exp_svm_geometry.png", dpi=140, bbox_inches="tight")
print("saved exp_svm_geometry")
