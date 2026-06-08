"""Robust PCA (principal component pursuit) через ADMM: low-rank + sparse.
Реальные кадры видеонаблюдения как столбцы матрицы M (низкоранг=фон, разреж=движение)."""
import numpy as np, matplotlib, matplotlib.pyplot as plt
from skimage import data
matplotlib.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':12})
# строим "видео": статичный реальный фон + перемещающийся реальный объект
rng=np.random.default_rng(0)
bg=data.camera().astype(float)/255.0
bg=bg[::4,::4]  # 128x128 уменьшим
h,w=bg.shape; T=40
obj=data.coins().astype(float)/255.0; obj=obj[40:80,40:80]; oh,ow=obj.shape
# круглая маска: объект — диск, а не квадратная заплатка (естественнее читается как «движущийся объект»)
yy,xx=np.mgrid[0:oh,0:ow]; cy,cx=(oh-1)/2,(ow-1)/2
mask=((yy-cy)**2+(xx-cx)**2) <= (min(oh,ow)/2-1)**2
frames=[]
for t in range(T):
    f=bg.copy()
    x0=int(8+(w-ow-16)*t/(T-1)); y0=h//2-oh//2
    patch=f[y0:y0+oh, x0:x0+ow]
    patch[mask]=obj[mask]  # реальный движущийся объект (круглый)
    frames.append(f.ravel())
M=np.stack(frames,axis=1)  # (h*w, T)
m,n=M.shape
lam=1.0/np.sqrt(max(m,n))
# ADMM / PCP
mu=0.25*m*n/np.sum(np.abs(M)); 
def svt(X,tau):
    U,s,Vt=np.linalg.svd(X,full_matrices=False); s=np.maximum(s-tau,0); return (U*s)@Vt
def soft(X,tau): return np.sign(X)*np.maximum(np.abs(X)-tau,0)
L=np.zeros_like(M); S=np.zeros_like(M); Y=np.zeros_like(M)
for _ in range(60):
    L=svt(M-S+Y/mu, 1/mu)
    S=soft(M-L+Y/mu, lam/mu)
    Y=Y+mu*(M-L-S)
r=np.linalg.matrix_rank(L,tol=1e-3)
print("rank(L)=",r,"||S||_0 frac=",np.mean(np.abs(S)>1e-3))
# показываем кадр t=20
t=20
fig,ax=plt.subplots(1,3,figsize=(10,3.6))
for a,im,ti in zip(ax,[M[:,t],L[:,t],np.abs(S[:,t])],
    ['Кадр $M$ (фон+объект)','Низкоранг $L$ (фон)','Разреж $|S|$ (движение)']):
    a.imshow(im.reshape(h,w),cmap='gray'); a.set_title(ti,fontsize=12); a.axis('off')
fig.suptitle(r'Robust PCA через ADMM: $\min\;\|L\|_*+\lambda\|S\|_1$ при $L+S=M$',y=1.02,fontsize=13)
fig.tight_layout(); fig.savefig('/root/hse26_repo/files/exp_admm_rpca.pdf',bbox_inches='tight'); fig.savefig('/tmp/exp_admm_rpca.png',bbox_inches='tight',dpi=135)
print("saved /tmp/exp_admm_rpca.png")
