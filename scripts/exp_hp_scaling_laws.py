"""
Hyperparameter scaling laws figure for lectures/19.md.

Plots the *published, fitted* closed-form laws (no invented data) for the
compute-optimal learning rate and batch size as functions of training compute C:

  DeepSeek LLM (Bi et al., 2024, arxiv 2401.02954), eq. fitted on real sweeps:
      eta_opt(C) = 0.3118 * C^(-0.1250)
      B_opt(C)   = 0.2920 * C^( 0.3271)

The markers show the compute budgets of a few real models for scale reference.
Grounded entirely on the published fit coefficients.

Saves files/hp_scaling_laws.{pdf,png}.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# DeepSeek LLM (2401.02954) fitted coefficients
def eta_opt(C):
    return 0.3118 * C ** (-0.1250)

def B_opt(C):
    return 0.2920 * C ** (0.3271)

C = np.logspace(17, 24, 200)   # training compute, FLOPs

plt.rcParams.update({
    'font.size': 13, 'axes.labelsize': 15, 'axes.titlesize': 15,
    'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

ax1.loglog(C, eta_opt(C), color='#2980b9', lw=3)
ax1.set_xlabel('Компьют $C$, FLOP')
ax1.set_ylabel(r'Оптимальный LR $\eta_{\mathrm{opt}}$')
ax1.set_title(r'$\eta_{\mathrm{opt}} = 0.3118\,C^{-0.125}$')
ax1.grid(True, which='both', alpha=0.25)

ax2.loglog(C, B_opt(C), color='#e74c3c', lw=3)
ax2.set_xlabel('Компьют $C$, FLOP')
ax2.set_ylabel(r'Оптимальный batch size $B_{\mathrm{opt}}$')
ax2.set_title(r'$B_{\mathrm{opt}} = 0.2920\,C^{0.327}$')
ax2.grid(True, which='both', alpha=0.25)

fig.suptitle('Гиперпараметрические scaling laws (DeepSeek LLM, 2401.02954): '
             'LR падает, batch растёт с компьютом', fontsize=13)
fig.text(0.995, 0.005, 'Формулы: DeepSeek 2401.02954 · @fminxyz', ha='right',
         va='bottom', color='gray', alpha=0.7, fontsize=10)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'files')
fig.savefig(os.path.join(out, 'hp_scaling_laws.pdf'), bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(out, 'hp_scaling_laws.png'), bbox_inches='tight', dpi=150)
print('eta_opt(1e20)=%.4g, B_opt(1e20)=%.4g' % (eta_opt(1e20), B_opt(1e20)))
print('saved to', out)
