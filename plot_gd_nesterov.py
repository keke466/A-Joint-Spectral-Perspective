# 文件名: plot_gd_nesterov.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 算法定义（同上，略）
def gradient_descent(x0, grad_func, L, T, f):
    x = x0.copy()
    fvals = [f(x)]
    step = 1 / L
    for _ in range(T):
        x = x - step * grad_func(x)
        fvals.append(f(x))
    return np.array(fvals)

def nesterov(x0, grad_func, L, T, f):
    x = x0.copy()
    y = x0.copy()
    t_prev = 1.0
    fvals = [f(x)]
    for k in range(1, T+1):
        x_new = y - (1.0 / L) * grad_func(y)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
        y = x_new + (t_prev - 1.0) / t_new * (x_new - x)
        x = x_new
        t_prev = t_new
        fvals.append(f(x))
    return np.array(fvals)

def generate_Sigma0(case, eig_H):
    d = len(eig_H)
    if case == 'isotropic':
        return np.eye(d)
    elif case == 'independent':
        sigma2 = np.random.uniform(0.1, 2.0, size=d)
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        return Q @ np.diag(sigma2) @ Q.T
    elif case == 'aligned':
        sigma2 = eig_H / np.mean(eig_H) * 2.0
        return np.diag(sigma2)
    elif case == 'misaligned':
        sigma2 = 1.0 / (eig_H + 0.1)
        sigma2 = sigma2 / np.mean(sigma2) * 2.0
        return np.diag(sigma2)

# 参数
d = 50
xi, tau = -0.5, 0.5
L = 10.0
T_plot = 500
n_avg = 20

np.random.seed(42)
eig_H = beta.rvs(a=xi+1, b=tau+1, size=d) * L
H = np.diag(eig_H)
L_max = np.max(eig_H)

def f(x): return 0.5 * x @ H @ x
def grad(x): return H @ x

cases = ['isotropic', 'independent', 'aligned', 'misaligned']
case_names = ['Isotropic', 'Independent', 'Aligned', 'Misaligned']
colors = {'isotropic': 'black', 'independent': 'blue', 'aligned': 'green', 'misaligned': 'red'}
linestyles = {'isotropic': '-', 'independent': '--', 'aligned': '-.', 'misaligned': ':'}

avg_curves = {algo: {case: None for case in cases} for algo in ['gd', 'nesterov']}

for case in cases:
    gd_curves, nes_curves = [], []
    for seed in range(n_avg):
        np.random.seed(seed)
        Sigma0 = generate_Sigma0(case, eig_H)
        x0 = np.random.multivariate_normal(np.zeros(d), Sigma0)
        gd_curves.append(gradient_descent(x0, grad, L_max, T_plot, f))
        nes_curves.append(nesterov(x0, grad, L_max, T_plot, f))
    min_len = min(len(c) for c in gd_curves)
    avg_curves['gd'][case] = np.mean([c[:min_len] for c in gd_curves], axis=0)
    min_len = min(len(c) for c in nes_curves)
    avg_curves['nesterov'][case] = np.mean([c[:min_len] for c in nes_curves], axis=0)

# 绘图
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
algo_names = {'gd': 'Gradient Descent', 'nesterov': 'Nesterov'}

for ax_idx, algo in enumerate(['gd', 'nesterov']):
    ax = axes[ax_idx]
    for case in cases:
        curve = avg_curves[algo][case]
        if curve is not None:
            iters = np.arange(len(curve))
            ax.semilogy(iters, curve, color=colors[case], linestyle=linestyles[case],
                        label=case_names[cases.index(case)], linewidth=1.2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$f(x)$')
    ax.set_title(algo_names[algo])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_ylim(1e-16, 1e2)

plt.tight_layout()
plt.savefig('gd_nesterov_convergence.pdf', dpi=300, bbox_inches='tight')
plt.show()