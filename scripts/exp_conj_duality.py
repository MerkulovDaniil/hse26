"""
Сопряжённая функция: двойственность «наклон <-> аргумент» (чистая замена старых conj_question/answer).
Слева:  f(x) и опорная прямая наклона y0, касающаяся графика в точке x0.
Справа: f*(y) — её наклон в точке y0 равен x0.
Связь:  y0 = f'(x0)  <=>  x0 = (f*)'(y0).  Субградиент одной = аргумент другой.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 14, 'axes.labelsize': 15, 'axes.titlesize': 15,
    'legend.fontsize': 12, 'font.family': 'serif', 'mathtext.fontset': 'cm',
})

f = lambda x: 0.4 * x ** 2 + 0.08 * x ** 4
fp = lambda x: 0.8 * x + 0.32 * x ** 3
x0 = 1.3
y0 = fp(x0)
f0 = f(x0)
fstar0 = x0 * y0 - f0                      # f*(y0) = x0 y0 - f(x0)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.7))

# ── Левая панель: f(x) + опорная прямая наклона y0 ──
xx = np.linspace(-2.1, 2.1, 400)
axL.plot(xx, f(xx), color='#C44E52', lw=2.6, label='$f(x)$')
line = f0 + y0 * (xx - x0)
axL.plot(xx, line, '--', color='#444', lw=1.6)
axL.plot([x0], [f0], 'o', color='k', ms=7, zorder=5)
axL.plot([x0, x0], [0, f0], ':', color='gray', lw=1)
axL.annotate(r'наклон $=y_0$', xy=(1.75, f0 + y0 * (1.75 - x0)), xytext=(0.1, 2.4),
             fontsize=13, color='#444',
             arrowprops=dict(arrowstyle='->', color='#444', lw=1))
axL.text(x0, -0.45, '$x_0$', ha='center', fontsize=14)
axL.set_xlabel('$x$'); axL.set_ylabel('$f(x)$')
axL.set_title('Опорная прямая наклона $y_0$ касается $f$ в $x_0$')
axL.set_ylim(-0.6, 3.2); axL.legend(loc='upper center')
axL.grid(alpha=0.25)

# ── Правая панель: f*(y) параметрически + наклон x0 в y0 ──
xs = np.linspace(-2.0, 2.0, 400)
Y = fp(xs)
Fstar = xs * fp(xs) - f(xs)
order = np.argsort(Y)
axR.plot(Y[order], Fstar[order], color='#4C72B0', lw=2.6, label='$f^*(y)$')
tang = fstar0 + x0 * (Y[order] - y0)
axR.plot(Y[order], tang, '--', color='#444', lw=1.6)
axR.plot([y0], [fstar0], 'o', color='k', ms=7, zorder=5)
axR.plot([y0, y0], [Fstar.min() - 0.2, fstar0], ':', color='gray', lw=1)
axR.annotate(r'наклон $=x_0$', xy=(y0 + 0.9, fstar0 + x0 * 0.9), xytext=(-1.9, 1.7),
             fontsize=13, color='#444',
             arrowprops=dict(arrowstyle='->', color='#444', lw=1))
axR.text(y0, Fstar.min() - 0.55, '$y_0$', ha='center', fontsize=14)
axR.set_xlabel('$y$'); axR.set_ylabel('$f^*(y)$')
axR.set_title('Наклон $f^*$ в точке $y_0$ равен $x_0$')
axR.set_ylim(-0.9, 5.9)   # тугой диапазон вокруг f*, чтобы 0 был внизу (как слева), а касательная просто клиппится
axR.legend(loc='upper center'); axR.grid(alpha=0.25)

fig.suptitle(r'Двойственность наклон $\leftrightarrow$ аргумент: $\;y_0=f\,\!\!\;^\prime(x_0)\ \Leftrightarrow\ x_0=(f^*)^\prime(y_0)$',
             y=1.02, fontsize=15)
fig.tight_layout()
fig.savefig('/root/hse26_repo/files/exp_conj_duality.pdf', bbox_inches='tight')
fig.savefig('/tmp/exp_conj_duality.png', bbox_inches='tight', dpi=140)
print('saved exp_conj_duality; x0=%.2f y0=%.2f f*(y0)=%.2f' % (x0, y0, fstar0))
