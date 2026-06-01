"""
REAL per-layer norms motivating LARS (lectures/18.md).

A genuine 12-layer MLP (He init, ReLU, softmax) trained for a few dozen SGD
steps on REAL data (sklearn digits, 1797 handwritten digits, 10 classes).
After training we do one forward/backward on a real batch and MEASURE, per
layer l:
  ||W_l||            weight norm
  ||grad_l||         gradient norm
  ||W_l|| / ||grad_l||   LARS "trust ratio"
  effective update under SGD (∝ ||grad_l||)  vs  LARS (∝ ||W_l||)

The point LARS makes: a single global learning rate is bad because the
weight/gradient ratio varies across layers by orders of magnitude; LARS
rescales per layer so every layer gets an update proportional to its own
weight norm. All numbers below are measured, not hand-drawn.

Saves files/lars_layerwise.{pdf,png}.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

# ---- REAL data: sklearn handwritten digits ----
d = load_digits()
X = StandardScaler().fit_transform(d.data)      # (1797, 64)
Y = np.eye(10)[d.target]                         # one-hot (1797, 10)
N, din = X.shape
dout = 10

# ---- 12-layer MLP, He init ----
dims = [din] + [128] * 11 + [dout]               # 12 weight matrices
L = len(dims) - 1
Ws = [np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i]) for i in range(L)]
bs = [np.zeros(dims[i + 1]) for i in range(L)]


def forward(Xb):
    acts, pre = [Xb], []
    h = Xb
    for i in range(L):
        z = h @ Ws[i] + bs[i]
        pre.append(z)
        h = np.maximum(0, z) if i < L - 1 else z
        acts.append(h)
    z = acts[-1] - acts[-1].max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return acts, pre, p


def backward(acts, pre, p, Yb):
    B = Yb.shape[0]
    grads = [None] * L
    delta = (p - Yb) / B                          # dL/dz at output
    for i in reversed(range(L)):
        grads[i] = acts[i].T @ delta
        if i > 0:
            delta = (delta @ Ws[i].T) * (pre[i - 1] > 0)
    return grads


# ---- train a few dozen real SGD steps ----
eta = 0.1
for step in range(40):
    idx = np.random.choice(N, 256, replace=False)
    acts, pre, p = forward(X[idx])
    grads = backward(acts, pre, p, Y[idx])
    for i in range(L):
        Ws[i] -= eta * grads[i]

# ---- measure per-layer norms on a real batch ----
idx = np.random.choice(N, 256, replace=False)
acts, pre, p = forward(X[idx])
grads = backward(acts, pre, p, Y[idx])
wn = np.array([np.linalg.norm(Ws[i]) for i in range(L)])
gn = np.array([np.linalg.norm(grads[i]) for i in range(L)])
coef = wn / gn                                    # LARS trust ratio
sgd_upd = eta * gn                                # update ∝ ||grad||
lars_upd = eta * wn                               # LARS: update ∝ ||w||
layers = np.arange(1, L + 1)
print("layer  ||w||      ||g||      w/g")
for i in range(L):
    print(f"{i+1:>4}  {wn[i]:.3e}  {gn[i]:.3e}  {coef[i]:.3e}")

# ---- plot ----
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
bw = 0.4

ax[0].bar(layers - bw / 2, wn, bw, color='#5aa0e8', label=r'$\|w_l\|$')
ax[0].bar(layers + bw / 2, gn, bw, color='#e8615a', label=r'$\|\nabla_l\|$')
ax[0].set_yscale('log'); ax[0].set_title('Нормы весов и градиентов')
ax[0].set_xlabel('Слой'); ax[0].set_ylabel('Норма (log)'); ax[0].legend()

ax[1].bar(layers, coef, color='#5ac77a')
ax[1].set_yscale('log'); ax[1].set_title('LARS: коэффициент масштабирования')
ax[1].set_xlabel('Слой'); ax[1].set_ylabel(r'$\|w_l\|/\|\nabla_l\|$ (log)')

ax[2].bar(layers - bw / 2, sgd_upd, bw, color='#e8615a', label='SGD')
ax[2].bar(layers + bw / 2, lars_upd, bw, color='#5ac77a', label='LARS')
ax[2].set_yscale('log'); ax[2].set_title('Эффективные обновления')
ax[2].set_xlabel('Слой'); ax[2].set_ylabel('Норма обновления (log)'); ax[2].legend()

for a in ax:
    a.set_xticks(layers)
    a.grid(True, axis='y', alpha=0.3)

fig.tight_layout()
fig.text(0.99, 0.005, '@fminxyz', ha='right', va='bottom', color='gray', alpha=0.5, fontsize=11)

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'files')
plt.savefig(os.path.join(outdir, 'lars_layerwise.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(outdir, 'lars_layerwise.png'), bbox_inches='tight', dpi=150)
print(f"Saved to {outdir}/lars_layerwise.pdf and .png")
