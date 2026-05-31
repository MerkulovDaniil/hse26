"""
Numerical experiment for the "Big models" lecture (lectures/19.md):
Gradient accumulation reproduces large-batch training exactly.

Setup: L2-regularized logistic regression (strongly convex), single fixed init.
Three runs, constant learning rate:
  1) B = 512                       (large batch)
  2) B = 64, accumulation x8       (effective 512) -> must coincide with (1)
  3) B = 64                        (small batch, no accumulation)

With mean-reduced loss and identical data partitioning, run (2) takes exactly
the same optimizer steps as run (1): the curves overlap up to numerical error.
Large batch reduces gradient noise -> lower noise floor than B=64.

Saves files/grad_accum_equivalence.{pdf,png}.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# ---- data: REAL handwritten digits (sklearn bundled, no synthetic data) ----
# Binary task: even vs odd digit. 1797 samples, 64 pixel features.
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)   # standardize 0..16 pixel intensities
y = np.where(digits.target % 2 == 0, 1, -1)       # even -> +1, odd -> -1
N, d = X.shape
mu = 1e-2  # L2 regularization -> strong convexity


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def loss(w, Xb, yb):
    margins = yb * (Xb @ w)
    return np.mean(np.log1p(np.exp(-np.clip(margins, -500, 500)))) + 0.5 * mu * w @ w


def grad(w, Xb, yb):
    margins = yb * (Xb @ w)
    s = (sigmoid(margins) - 1) * yb           # dL/d(margin)
    return Xb.T @ s / len(yb) + mu * w


def full_loss(w):
    return loss(w, X, y)


# ---- optimum (for f - f*) ----
w_star = minimize(full_loss, np.zeros(d), jac=lambda w: grad(w, X, y),
                  method='L-BFGS-B', options={'maxiter': 2000}).x
f_star = full_loss(w_star)

w0 = np.zeros(d)
lr = 0.3
n_passes = 60            # passes over the dataset
B_large, ACC = 256, 8    # effective batch = 256 (= 32 x 8)
rng = np.random.default_rng(123)


def run(batch, accum, lr):
    """One SGD trajectory. Records f(w)-f* after every pass over the data."""
    w = w0.copy()
    hist = [full_loss(w) - f_star]
    steps_per_pass = N // (batch * accum)
    for _ in range(n_passes):
        perm = rng.permutation(N)
        ptr = 0
        for _ in range(steps_per_pass):
            g = np.zeros(d)
            for _ in range(accum):                       # accumulate micro-batches
                idx = perm[ptr:ptr + batch]; ptr += batch
                g += grad(w, X[idx], y[idx])
            w -= lr * g / accum                          # mean over the effective batch
        hist.append(full_loss(w) - f_star)
    return np.maximum(np.array(hist), 1e-16)


# same RNG seed per run so the data partition is identical for (1) and (2)
rng = np.random.default_rng(123); h_large = run(batch=B_large,       accum=1,   lr=lr)
rng = np.random.default_rng(123); h_accum = run(batch=B_large // ACC, accum=ACC, lr=lr)
rng = np.random.default_rng(7);   h_small = run(batch=B_large // ACC, accum=1,   lr=lr)

print(f"max |B={B_large} - (B={B_large//ACC} accum x{ACC})| = {np.max(np.abs(h_large - h_accum)):.2e}")
print(f"final: B={B_large} {h_large[-1]:.2e}, accum {h_accum[-1]:.2e}, B={B_large//ACC} {h_small[-1]:.2e}")

# ---- plot ----
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16,
    'legend.fontsize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
})
fig, ax = plt.subplots(1, 1, figsize=(8, 5.6))
passes = np.arange(n_passes + 1)
ax.semilogy(passes, h_small, '-', color='#e74c3c', linewidth=2.0, alpha=0.7,
            label=rf'$B={B_large//ACC}$ (малый батч)')
ax.semilogy(passes, h_large, '-', color='#2980b9', linewidth=3.0, alpha=0.9,
            label=rf'$B={B_large}$ (большой батч)')
ax.semilogy(passes, h_accum, '--', color='#f39c12', linewidth=2.2,
            label=rf'$B={B_large//ACC}$, accum $\times {ACC}$')
ax.set_xlabel('Число проходов по данным')
ax.set_ylabel(r'$f(w^k) - f^*$')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_title('Аккумуляция градиентов воспроизводит большой батч')
fig.text(0.99, 0.01, '@fminxyz', ha='right', va='bottom', color='gray', alpha=0.5, fontsize=12)

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'files')
plt.savefig(os.path.join(outdir, 'grad_accum_equivalence.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(outdir, 'grad_accum_equivalence.png'), bbox_inches='tight', dpi=150)
print(f"Saved to {outdir}/grad_accum_equivalence.pdf and .png")
