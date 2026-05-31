"""
Regenerate the "trends" figures for lectures/19.md from the *current*
Epoch AI "Notable AI Models" dataset (https://epoch.ai/data/notable-ai-models).

Grounded strictly on real data: reads scripts/epoch_notable_models.csv
(snapshot downloaded from epoch.ai). Produces, in files/:
  - compute_trends_global.pdf : training compute vs date, full history (1950+)
  - compute_trends_local.pdf  : training compute vs date, deep-learning era (2010+)
  - num_param_trends.pdf      : trainable parameters vs date (2010+)

Re-download the snapshot with:
  curl -sL -o scripts/epoch_notable_models.csv https://epoch.ai/data/notable_ai_models.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(HERE, 'epoch_notable_models.csv')
OUT = os.path.join(ROOT, 'files')

df = pd.read_csv(CSV)
df['date'] = pd.to_datetime(df['Publication date'], errors='coerce')
df['compute'] = pd.to_numeric(df['Training compute (FLOP)'], errors='coerce')
df['params'] = pd.to_numeric(df['Parameters'], errors='coerce')


def main_domain(s):
    """Collapse Epoch's multi-label domains to a single primary bucket."""
    if not isinstance(s, str):
        return 'Other'
    parts = [p.strip() for p in s.split(',')]
    if len(parts) > 1 or 'Multimodal' in parts:
        return 'Multimodal'
    return parts[0]


df['domain'] = df['Domain'].apply(main_domain)

# Epoch-like categorical palette
COLORS = {
    'Language': '#1f9e89', 'Vision': '#e07b39', 'Multimodal': '#6a4c93',
    'Image generation': '#3b6fb6', 'Speech': '#c0392b', 'Games': '#8e44ad',
    'Biology': '#16a085', 'Robotics': '#2c3e50', 'Video': '#d4a017',
    'Recommendation': '#7f8c8d', 'Other': '#95a5a6',
}
ORDER = ['Language', 'Vision', 'Multimodal', 'Image generation', 'Speech',
         'Games', 'Biology', 'Robotics', 'Video', 'Recommendation', 'Other']

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'legend.fontsize': 9, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
})


def scatter_by_domain(ax, d, ycol):
    for dom in ORDER:
        sub = d[d['domain'] == dom]
        if len(sub):
            ax.scatter(sub['date'], sub[ycol], s=16, alpha=0.65,
                       color=COLORS.get(dom, '#95a5a6'), label=dom,
                       edgecolors='none', rasterized=True)


def fit_growth(d, ycol, t0):
    """Exponential fit log10(y) ~ a + b*years; returns (years_x, yhat, x_per_year)."""
    dd = d.dropna(subset=[ycol, 'date'])
    yrs = (dd['date'] - t0).dt.days.values / 365.25
    ly = np.log10(dd[ycol].values)
    b, a = np.polyfit(yrs, ly, 1)
    xs = np.linspace(yrs.min(), yrs.max(), 100)
    return t0 + pd.to_timedelta(xs * 365.25, unit='D'), 10 ** (a + b * xs), 10 ** b


def style(ax):
    ax.set_yscale('log')
    ax.grid(True, which='major', alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


# ---------- 1) compute, full history ----------
d = df.dropna(subset=['compute', 'date'])
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.axvspan(pd.Timestamp('2010-01-01'), d['date'].max() + pd.Timedelta(days=200),
           color='#dbe7f3', alpha=0.5, zorder=0)
ax.text(pd.Timestamp('2017-06-01'), d['compute'].min() * 2, 'Эра глубокого обучения',
        color='#5d7fa6', fontsize=11, ha='center')
scatter_by_domain(ax, d, 'compute')
gx, gy, rate = fit_growth(d, 'compute', d['date'].min())
ax.plot(gx, gy, 'k--', lw=1.6, alpha=0.8)
ax.text(0.04, 0.9, f'тренд: $\\times${rate:.1f}/год', transform=ax.transAxes,
        fontsize=11, color='black')
ax.set_ylabel('Training compute (FLOP)')
ax.set_title(f'Notable AI Models — обучающие вычисления ({len(d)} моделей)')
ax.legend(loc='lower right', ncol=2, framealpha=0.9, markerscale=1.4)
fig.text(0.995, 0.004, 'Данные: epoch.ai · @fminxyz', ha='right', va='bottom',
         color='gray', alpha=0.7, fontsize=10)
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'compute_trends_global.pdf'), bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUT, 'compute_trends_global.png'), bbox_inches='tight', dpi=150)
plt.close(fig)

# ---------- 2) compute, deep-learning era ----------
d2 = d[d['date'] >= '2010-01-01']
fig, ax = plt.subplots(figsize=(11, 6.2))
scatter_by_domain(ax, d2, 'compute')
gx, gy, rate = fit_growth(d2, 'compute', d2['date'].min())
ax.plot(gx, gy, 'k--', lw=1.8, alpha=0.85)
ax.text(0.04, 0.92, f'тренд (2010+): $\\times${rate:.1f}/год', transform=ax.transAxes,
        fontsize=12, color='black')
# annotate a few frontier models
top = d2.sort_values('compute').tail(5)
for i, (_, r) in enumerate(top.iterrows()):
    ax.annotate(str(r['Model']), (r['date'], r['compute']), fontsize=8,
                xytext=(-6, 7 + 9 * (i % 3)), textcoords='offset points',
                ha='right', alpha=0.85)
ax.set_ylabel('Training compute (FLOP)')
ax.set_title(f'Notable AI Models — эра глубокого обучения ({len(d2)} моделей)')
ax.legend(loc='lower right', ncol=2, framealpha=0.9, markerscale=1.4)
fig.text(0.995, 0.004, 'Данные: epoch.ai · @fminxyz', ha='right', va='bottom',
         color='gray', alpha=0.7, fontsize=10)
ax.set_yscale('log'); ax.grid(True, which='major', alpha=0.25)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'compute_trends_local.pdf'), bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUT, 'compute_trends_local.png'), bbox_inches='tight', dpi=150)
plt.close(fig)

# ---------- 3) parameters, deep-learning era ----------
dp = df.dropna(subset=['params', 'date'])
dp = dp[dp['date'] >= '2010-01-01']
fig, ax = plt.subplots(figsize=(11, 6.2))
scatter_by_domain(ax, dp, 'params')
gx, gy, rate = fit_growth(dp, 'params', dp['date'].min())
ax.plot(gx, gy, 'k--', lw=1.8, alpha=0.85)
ax.text(0.04, 0.92, f'тренд (2010+): $\\times${rate:.1f}/год', transform=ax.transAxes,
        fontsize=12, color='black')
ax.set_ylabel('Число обучаемых параметров')
ax.set_title(f'Notable AI Models — число параметров ({len(dp)} моделей)')
ax.legend(loc='lower right', ncol=2, framealpha=0.9, markerscale=1.4)
fig.text(0.995, 0.004, 'Данные: epoch.ai · @fminxyz', ha='right', va='bottom',
         color='gray', alpha=0.7, fontsize=10)
ax.set_yscale('log'); ax.grid(True, which='major', alpha=0.25)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'num_param_trends.pdf'), bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUT, 'num_param_trends.png'), bbox_inches='tight', dpi=150)
plt.close(fig)

print('compute (global):', len(d), '| compute (2010+):', len(d2), '| params (2010+):', len(dp))
print('latest date in data:', df['date'].max().date())
print('saved 3 figures to', OUT)
