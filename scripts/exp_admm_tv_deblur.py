"""
Реальная задача, где ADMM ценен СТРУКТУРОЙ (не скоростью): TV-ДЕБЛЮРИНГ фото.
Восстанавливаем РЕЗКОЕ изображение из смазанного и зашумлённого (skimage 'camera'):
    min_x  1/2 || K x - f ||^2 + lambda * TV(x),   K — оператор размытия (свёртка).
Честно: по PSNR ADMM и FISTA сопоставимы (FISTA даже чуть выше). Ценность ADMM здесь иная:
  * ADMM решает ТОЧНУЮ негладкую TV-задачу; FISTA минимизирует СГЛАЖЕННЫЙ суррогат и смещён по J
    (плато по функционалу из-за сглаживания, Jgap ~ 0.6).
  * Оператор размытия K плохо обусловлен; ADMM в x-шаге решает (K^T K + rho D^T D) x = ... ТОЧНО
    одним FFT — замкнутые шаги, не чувствительные к обусловленности K.
ADMM (split Bregman): x-шаг = деление в Фурье; z-шаг = изотропная усадка (прокс TV); u-шаг.
FISTA: ускоренный прокс-градиент на СГЛАЖЕННОМ TV, шаг 1/L (L от K и сглаживания).
Сравнение по PSNR(дБ) на каждой итерации — кто быстрее восстанавливает резкость. Реальные данные.
Воспроизводимо: фиксированный seed.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fft import fft2, ifft2
from skimage import data

matplotlib.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12.5,
    'legend.fontsize': 11, 'font.family': 'serif', 'mathtext.fontset': 'cm',
})

rng = np.random.default_rng(0)
clean = data.camera().astype(float) / 255.0
m, n = clean.shape

# ── Оператор размытия K (гауссов PSF) в Фурье ──
ksz, sig = 15, 1.8
ax = np.arange(ksz) - ksz // 2
g = np.exp(-ax ** 2 / (2 * sig ** 2)); ker = np.outer(g, g); ker /= ker.sum()
psf = np.zeros((m, n)); psf[:ksz, :ksz] = ker
psf = np.roll(psf, (-(ksz // 2), -(ksz // 2)), (0, 1))     # центр PSF в (0,0)
Khat = fft2(psf)
def Kop(x):  return np.real(ifft2(Khat * fft2(x)))
def KTop(x): return np.real(ifft2(np.conj(Khat) * fft2(x)))

blurry = Kop(clean) + 0.01 * rng.standard_normal((m, n))
lam = 0.003

def Dh(x): return np.roll(x, -1, 1) - x
def Dv(x): return np.roll(x, -1, 0) - x
def DhT(p): return np.roll(p, 1, 1) - p
def DvT(p): return np.roll(p, 1, 0) - p
def psnr(x): return 10 * np.log10(1.0 / np.mean((np.clip(x, 0, 1) - clean) ** 2))
def Jtv(x): return 0.5 * np.sum((Kop(x) - blurry) ** 2) + lam * np.sum(np.sqrt(Dh(x) ** 2 + Dv(x) ** 2))

kh = 2 - 2 * np.cos(2 * np.pi * np.arange(n) / n)
kv = 2 - 2 * np.cos(2 * np.pi * np.arange(m) / m)
lap_hat = kv[:, None] + kh[None, :]
K2 = np.abs(Khat) ** 2

K = 150
# ── ADMM (split Bregman) ──
rho = 0.05
denom = K2 + rho * lap_hat
KTf = KTop(blurry)
x = blurry.copy(); zh = Dh(x); zv = Dv(x); uh = np.zeros_like(x); uv = np.zeros_like(x)
admm_psnr, Jadmm = [], []
for _ in range(K):
    rhs = KTf + rho * (DhT(zh - uh) + DvT(zv - uv))      # K^T f + rho D^T(z-u)
    x = np.real(ifft2(fft2(rhs) / denom))
    ah, av = Dh(x) + uh, Dv(x) + uv
    mag = np.sqrt(ah ** 2 + av ** 2)
    sh = np.maximum(1 - lam / (rho * np.maximum(mag, 1e-12)), 0.0)
    zh, zv = sh * ah, sh * av
    uh += Dh(x) - zh; uv += Dv(x) - zv
    admm_psnr.append(psnr(x)); Jadmm.append(Jtv(x))
x_admm = np.clip(x, 0, 1)

# ── FISTA на сглаженном TV ──
eps = 0.01
L = np.max(K2) + lam * 8.0 / eps
step = 1.0 / L
xf = blurry.copy(); y = xf.copy(); t = 1.0; fista_psnr, Jfista = [], []
for _ in range(K):
    gh, gv = Dh(y), Dv(y)
    w = np.sqrt(gh ** 2 + gv ** 2 + eps ** 2)
    grad = KTop(Kop(y) - blurry) + lam * (DhT(gh / w) + DvT(gv / w))
    xn = y - step * grad
    tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
    y = xn + ((t - 1) / tn) * (xn - xf); t = tn; xf = xn
    fista_psnr.append(psnr(xf)); Jfista.append(Jtv(xf))
x_fista = np.clip(xf, 0, 1)

fstar = min(min(Jadmm), min(Jfista))
print(f'blurry={psnr(blurry):.2f}  ADMM={admm_psnr[-1]:.2f}/{admm_psnr[19]:.2f}@20  FISTA={fista_psnr[-1]:.2f} dB')
print(f'J*: {fstar:.4f}  ADMM Jgap={Jadmm[-1]-fstar:.2e}  FISTA Jgap(плато,смещение)={Jfista[-1]-fstar:.2e}')

# ── Рисунок: 3 панели (смазано → ADMM → сходимость по функционалу) ──
fig = plt.figure(figsize=(13, 4.3))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.55], wspace=0.22)
def show(ax, im, title):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1); ax.set_title(title, fontsize=12.5)
    ax.set_xticks([]); ax.set_yticks([])
show(fig.add_subplot(gs[0, 0]), blurry, f'Смазано $+$ шум  ({psnr(blurry):.1f} дБ)')
show(fig.add_subplot(gs[0, 1]), x_admm, f'ADMM  ({admm_psnr[-1]:.1f} дБ)')

axc = fig.add_subplot(gs[0, 2])
it = np.arange(1, K + 1)
axc.semilogy(it, np.maximum(np.array(Jadmm) - fstar, 1e-12), color='#C44E52', lw=2.6,
             label='ADMM (точный TV)')
axc.semilogy(it, np.maximum(np.array(Jfista) - fstar, 1e-12), color='#55A868', lw=2.2,
             label='FISTA (сглаженный TV)')
gapf = Jfista[-1] - fstar
axc.axhline(gapf, ls=':', color='#55A868', lw=1.2)
axc.text(K * 0.50, gapf * 1.5, 'плато FISTA:\nсмещение сглаживания',
         fontsize=10.5, color='#3c7a4f', ha='center', va='bottom')
axc.set_xlabel('итерация $k$', fontsize=13); axc.set_ylabel(r'$J(x_k) - J^\star$', fontsize=13)
axc.set_title('Сходимость по функционалу $J$', fontsize=13)
axc.legend(loc='lower left'); axc.grid(alpha=0.3, which='both')

fig.savefig('/root/hse26_repo/files/exp_admm_tv_deblur.pdf', bbox_inches='tight')
fig.savefig('/tmp/exp_admm_tv_deblur.png', bbox_inches='tight', dpi=135)
print('saved exp_admm_tv_deblur')
