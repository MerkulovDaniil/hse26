"""
Bit-layout diagram of floating-point formats for the "Mixed precision" slide
(lectures/18.md). Minimalist, Russian labels — shows how each format splits its
bits into sign / exponent (range) / mantissa (precision).

Insight made visual:
  TF32 = FP32 exponent (8) + FP16 mantissa (10)   -> FP32 range, FP16 precision
  BF16 vs FP16: BF16 has wider exponent (range), narrower mantissa (precision)

Saves files/float_formats_bits.{pdf,png}.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# (name, sign, exponent, mantissa)
FORMATS = [
    ("FP32",       1, 8, 23),
    ("TF32",       1, 8, 10),
    ("FP16",       1, 5, 10),
    ("BF16",       1, 8,  7),
    ("FP8 (E4M3)", 1, 4,  3),
]

C_SIGN = "#cfe8ff"
C_EXP  = "#b6efad"
C_MANT = "#f7b3b3"
EDGE   = "black"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16,
})
fig, ax = plt.subplots(1, 1, figsize=(10, 4.4))

bar_h = 0.62
y_top = len(FORMATS) - 1

for i, (name, s, e, m) in enumerate(FORMATS):
    y = y_top - i
    total = s + e + m
    # segments
    ax.add_patch(Rectangle((0, y - bar_h / 2), s, bar_h, fc=C_SIGN, ec=EDGE, lw=1.3))
    ax.add_patch(Rectangle((s, y - bar_h / 2), e, bar_h, fc=C_EXP, ec=EDGE, lw=1.3))
    ax.add_patch(Rectangle((s + e, y - bar_h / 2), m, bar_h, fc=C_MANT, ec=EDGE, lw=1.3))
    # bit counts inside exponent / mantissa
    ax.text(s + e / 2, y, str(e), ha="center", va="center", fontsize=13)
    ax.text(s + e + m / 2, y, str(m), ha="center", va="center", fontsize=13)
    # format name on the left
    ax.text(-0.6, y, name, ha="right", va="center", fontsize=14, fontweight="bold")
    # total bits on the right
    ax.text(total + 0.6, y, f"{total} бит", ha="left", va="center",
            fontsize=12, color="#444")

# legend with the essential mapping
handles = [
    Patch(fc=C_SIGN, ec=EDGE, label="знак"),
    Patch(fc=C_EXP,  ec=EDGE, label="экспонента — диапазон"),
    Patch(fc=C_MANT, ec=EDGE, label="мантисса — точность"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=3, frameon=False, fontsize=13, handlelength=1.3, columnspacing=1.6)

ax.set_xlim(-9, 41)
ax.set_ylim(-0.7, y_top + 0.8)
ax.axis("off")
fig.text(0.99, 0.01, "@fminxyz", ha="right", va="bottom",
         color="gray", alpha=0.5, fontsize=12)

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "files")
plt.savefig(os.path.join(outdir, "float_formats_bits.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(outdir, "float_formats_bits.png"), bbox_inches="tight", dpi=150)
print(f"Saved to {outdir}/float_formats_bits.pdf and .png")
