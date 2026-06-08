"""Эволюция методов на LASSO: субградиент -> проксимальный -> ускоренный -> ADMM.
min 1/2||Ax-b||^2 + lam||x||_1. Показывает, ПОЧЕМУ курс шёл к расщеплению."""
import numpy as np, matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':13,
  'axes.labelsize':14,'axes.titlesize':14,'legend.fontsize':12})
rng=np.random.default_rng(0); m,n,k=200,500,25
A=rng.standard_normal((m,n))/np.sqrt(m); xt=np.zeros(n); s=rng.choice(n,k,False); xt[s]=rng.standard_normal(k)
b=A@xt+0.01*rng.standard_normal(m); lam=0.1*np.max(np.abs(A.T@b))
soft=lambda v,t:np.sign(v)*np.maximum(np.abs(v)-t,0); obj=lambda x:0.5*np.sum((A@x-b)**2)+lam*np.sum(np.abs(x))
AtA,Atb=A.T@A,A.T@b; L=np.linalg.norm(A,2)**2; N=400
# субградиент (diminishing step ~ c/sqrt(k))
x=np.zeros(n); hs=[]; c=1.0/L
for kk in range(1,N+1):
    g=A.T@(A@x-b)+lam*np.sign(x); x=x-(c/np.sqrt(kk))*g; hs.append(obj(x))
# ISTA
x=np.zeros(n); hi=[]
for _ in range(N): x=soft(x-(A.T@(A@x-b))/L,lam/L); hi.append(obj(x))
# FISTA
x=np.zeros(n); y=x.copy(); t=1; hf=[]
for _ in range(N):
    xn=soft(y-(A.T@(A@y-b))/L,lam/L); tn=(1+np.sqrt(1+4*t*t))/2; y=xn+((t-1)/tn)*(xn-x); t=tn; x=xn; hf.append(obj(x))
# ADMM
rho=1.0; Ch=np.linalg.cholesky(AtA+rho*np.eye(n)); x=np.zeros(n); z=np.zeros(n); u=np.zeros(n); ha=[]
for _ in range(N):
    x=np.linalg.solve(Ch.T,np.linalg.solve(Ch,Atb+rho*(z-u))); z=soft(x+u,lam/rho); u=u+x-z; ha.append(obj(x))
fs=min(min(hs),min(hi),min(hf),min(ha)); gap=lambda h:np.maximum(np.array(h)-fs,1e-16); it=np.arange(1,N+1)
fig,ax=plt.subplots(figsize=(7.5,4.6))
ax.loglog(it,gap(hs),color='#999',lw=2,label='субградиентный  $O(1/\\sqrt{k})$')
ax.loglog(it,gap(hi),color='#4C72B0',lw=2,label='проксимальный (ISTA)  $O(1/k)$')
ax.loglog(it,gap(hf),color='#55A868',lw=2,label='ускоренный (FISTA)  $O(1/k^2)$')
ax.loglog(it,gap(ha),color='#C44E52',lw=2.6,label='ADMM (расщепление)')
ax.set_xlabel('итерация $k$'); ax.set_ylabel(r'$f(x_k)-f^\star$')
ax.set_title('LASSO: эволюция методов курса'); ax.legend(); ax.grid(alpha=0.3,which='both')
fig.tight_layout(); fig.savefig('/root/hse26_repo/files/exp_methods_evolution.pdf',bbox_inches='tight'); fig.savefig('/tmp/exp_methods_evolution.png',bbox_inches='tight',dpi=140)
print('saved; subgrad final gap=%.2e ISTA=%.2e FISTA=%.2e ADMM=%.2e'%(gap(hs)[-1],gap(hi)[-1],gap(hf)[-1],gap(ha)[-1]))
