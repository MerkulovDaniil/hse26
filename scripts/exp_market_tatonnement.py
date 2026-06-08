import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.size"] = 11

# palette
C_MARKET = "#DD8452"  # market price (orange)
C_DUAL = "#4C72B0"    # demand / dual (blue)

rng = np.random.default_rng(3)
B = 5
# Параметры подобраны так, чтобы равновесная цена была ПОЛОЖИТЕЛЬНОЙ, а старт (lambda=0)
# шёл из ДЕФИЦИТА: спрос > предложения -> цена растёт, спрос падает к предложению.
a = rng.uniform(0.6, 2.0, B)
b = rng.uniform(2.0, 3.4, B)
total = 4.0

# closed form demand of agent i: x_i(lambda) = (b_i - lambda)/a_i
def demand(lam):
    return (b - lam) / a

def total_demand(lam):
    return demand(lam).sum()

# equilibrium price (closed form): sum (b_i - lam)/a_i = total
# => sum b_i/a_i - lam sum 1/a_i = total
inv_a_sum = (1.0 / a).sum()
lam_star = (np.sum(b / a) - total) / inv_a_sum

# dual ascent (tatonnement)
alpha = 0.4 / inv_a_sum  # under-relaxation so dynamics are visible
K = 30
lam = 0.0
lam_hist = [lam]
d_hist = [total_demand(lam)]
for k in range(K):
    d = total_demand(lam)
    lam = lam + alpha * (d - total)
    lam_hist.append(lam)
    d_hist.append(total_demand(lam))

lam_hist = np.array(lam_hist)
d_hist = np.array(d_hist)
it = np.arange(len(lam_hist))

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7.2, 5.6), sharex=True,
    gridspec_kw={"hspace": 0.12},
)

# regime shading based on demand d_k vs supply
def shade(ax):
    for k in range(len(d_hist) - 1):
        x0, x1 = it[k] - 0.5, it[k] + 0.5
        if d_hist[k] > total:
            ax.axvspan(x0, x1, color="#C44E52", alpha=0.10, lw=0)
        elif d_hist[k] < total:
            ax.axvspan(x0, x1, color="#4C72B0", alpha=0.08, lw=0)

# top: price
shade(ax_top)
ax_top.axhline(lam_star, ls="--", color="0.35", lw=1.3,
               label=r"$\lambda^* = %.3f$" % lam_star)
ax_top.plot(it, lam_hist, "o-", color=C_MARKET, ms=4, lw=1.8,
            label=r"цена $\lambda_k$")
ax_top.set_ylabel(r"цена $\lambda_k$")
ax_top.legend(loc="lower right", framealpha=0.9)

# bottom: total demand
shade(ax_bot)
ax_bot.axhline(total, ls="--", color="0.35", lw=1.3,
               label=r"предложение $= %.0f$" % total)
ax_bot.plot(it, d_hist, "o-", color=C_DUAL, ms=4, lw=1.8,
            label=r"спрос $d_k = \sum_i x_i(\lambda_k)$")
ax_bot.set_ylabel(r"спрос $d_k$")
ax_bot.set_xlabel("итерация $k$")
ax_bot.legend(loc="upper right", framealpha=0.9)

# legend proxies for regimes (only those actually present on the plot)
from matplotlib.patches import Patch
has_over = bool((d_hist[:-1] > total).any())
has_under = bool((d_hist[:-1] < total).any())
reg_handles = []
if has_over:
    reg_handles.append(Patch(color="#C44E52", alpha=0.10,
                             label=r"перегруз ($d_k>%.0f$): цена растёт" % total))
if has_under:
    reg_handles.append(Patch(color="#4C72B0", alpha=0.08,
                             label=r"недогруз ($d_k<%.0f$): цена падает" % total))
ax_top.add_artist(ax_top.legend(handles=reg_handles, loc="upper right",
                                framealpha=0.9, fontsize=9))
# restore main top legend (add_artist consumed default); re-add
ax_top.legend(
    handles=[
        plt.Line2D([], [], color=C_MARKET, marker="o", ms=4, lw=1.8,
                   label=r"цена $\lambda_k$"),
        plt.Line2D([], [], color="0.35", ls="--", lw=1.3,
                   label=r"$\lambda^* = %.3f$" % lam_star),
    ],
    loc="lower right", framealpha=0.9,
)

ax_top.set_title("Рыночное нащупывание цены")
ax_top.margins(x=0.01)
ax_bot.margins(x=0.01)

fig.savefig("/root/hse26_repo/files/exp_market_tatonnement.pdf",
            bbox_inches="tight")
fig.savefig("/tmp/exp_market_tatonnement.png", dpi=140, bbox_inches="tight")
print("lam_star=%.4f  final lam=%.4f  final d=%.4f" %
      (lam_star, lam_hist[-1], d_hist[-1]))
print("alpha=%.4f  inv_a_sum=%.4f" % (alpha, inv_a_sum))
