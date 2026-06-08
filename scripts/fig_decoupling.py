"""Анатомия блочной структуры для двойственной декомпозиции.
Матрица ограничений A разбивается на B блоков по столбцам: A=[A_1|...|A_B];
вектор x — на B блоков: x=[x_1;...;x_B]. Тогда Ax = sum_i A_i x_i = b.
Показываем, что даёт разбиение: целевая функция Σ f_i(x_i) разделима по блокам,
а единственная связь между блоками — общий ресурс Σ A_i x_i = b.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13})

B = 5
cols = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#937860']  # 5 блоков, без жёлтого
C_LINK = '#C44E52'

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

# ───────── блочная матрица A = [A_1 | ... | A_B] ─────────
Ax0, Ay0, Aw, Ah = 0.6, 3.1, 4.4, 2.1
sw = Aw / B
for i in range(B):
    ax.add_patch(Rectangle((Ax0 + i * sw, Ay0), sw, Ah,
                 facecolor=cols[i], edgecolor='0.25', lw=1.0, alpha=0.85))
    ax.text(Ax0 + (i + 0.5) * sw, Ay0 + Ah + 0.16, fr"$A_{i+1}$",
            ha='center', va='bottom', fontsize=12.5, color=cols[i])
    ax.plot([Ax0 + (i + 1) * sw] * 2, [Ay0, Ay0 + Ah], color='white', lw=1.0)
ax.text(Ax0 + Aw / 2, Ay0 - 0.32, r"$A$ ($m\times n$), столбцы по блокам",
        ha='center', va='top', fontsize=11, color='0.3')

# ───────── вектор x = [x_1; ...; x_B] ─────────
vx0, vw = Ax0 + Aw + 0.45, 0.55
seg = Ah / B
for i in range(B):
    yy = Ay0 + Ah - (i + 1) * seg
    ax.add_patch(Rectangle((vx0, yy), vw, seg,
                 facecolor=cols[i], edgecolor='0.25', lw=1.0, alpha=0.85))
    ax.text(vx0 + vw + 0.14, yy + seg / 2, fr"$x_{i+1}$",
            ha='left', va='center', fontsize=11.5, color=cols[i])
ax.text(vx0 + vw / 2, Ay0 - 0.32, r"$x$", ha='center', va='top', fontsize=11, color='0.3')

# ───────── = b ─────────
eqx = vx0 + vw + 1.05
ax.text(eqx, Ay0 + Ah / 2, r"$=$", ha='center', va='center', fontsize=20)
bx0 = eqx + 0.4
ax.add_patch(Rectangle((bx0, Ay0), 0.55, Ah, facecolor='0.82', edgecolor='0.25', lw=1.0))
ax.text(bx0 + 0.275, Ay0 - 0.32, r"$b$ ($m\times1$)", ha='center', va='top', fontsize=11, color='0.3')

# пояснение справа
ax.text(8.7, 4.7,
        "разбили на $B$ блоков:\n"
        r"столбцы $A$ и строки $x$",
        fontsize=11.5, va='top', ha='left', color='0.3')

# ───────── разложение произведения: Ax = Σ A_i x_i = b ─────────
y_sum = 1.95
ax.text(0.6, y_sum, r"$Ax=$", fontsize=15, va='center')
slot0, dx = 1.95, 1.7
for i in range(B):
    xc = slot0 + i * dx
    ax.text(xc, y_sum, fr"$A_{i+1}x_{i+1}$", color=cols[i], fontsize=15,
            va='center', ha='center')
    if i < B - 1:
        ax.text(xc + dx / 2, y_sum, r"$+$", fontsize=15, va='center', ha='center')
ax.text(slot0 + (B - 1) * dx + dx * 0.62, y_sum, r"$=\,b$", fontsize=15, va='center', ha='left')

# ───────── два вывода ─────────
ax.text(0.6, 1.05,
        r"целевая функция разделима:  $f(x)=\sum_{i=1}^B f_i(x_i)$  — каждый $x_i$ только в своём $f_i$",
        fontsize=13, va='center', color='0.15')
ax.text(0.6, 0.4,
        r"единственная связь блоков — общий ресурс  $\sum_{i=1}^B A_i x_i = b$",
        fontsize=13, va='center', color=C_LINK)

fig.tight_layout()
fig.savefig("/root/hse26_repo/files/decoupling.png", dpi=150, bbox_inches="tight")
fig.savefig("/tmp/decoupling.png", dpi=150, bbox_inches="tight")
print("saved decoupling, B=%d" % B)
