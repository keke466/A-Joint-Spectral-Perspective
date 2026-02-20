# -*- coding: utf-8 -*-
"""
实验一：四种初始化类型对比（GD 与 Nesterov）
运行后生成：
- 收敛曲线图（gd_nesterov_four_cases.pdf）
- 统计表格（four_cases_stats.csv）
- 每次试验的原始数据（four_cases_raw.csv）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta, bootstrap, mannwhitneyu

# ================== 算法定义 ==================
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

# ================== 初始化协方差生成 ==================
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
    else:
        raise ValueError(f"Unknown case: {case}")

# ================== 实验参数 ==================
d = 50
xi, tau = -0.5, 0.5
L = 10.0
T_max = 2000
n_avg = 50                     # 试验次数（增加以稳定统计）
threshold = 1e-6

np.random.seed(42)
eig_H = beta.rvs(a=xi+1, b=tau+1, size=d) * L
H = np.diag(eig_H)
L_max = np.max(eig_H)

def f(x): return 0.5 * x @ H @ x
def grad(x): return H @ x

cases = ['isotropic', 'independent', 'aligned', 'misaligned']
case_names = ['Isotropic', 'Independent', 'Aligned', 'Misaligned']
colors = {'isotropic': 'black', 'independent': 'blue',
          'aligned': 'green', 'misaligned': 'red'}
linestyles = {'isotropic': '-', 'independent': '--',
              'aligned': '-.', 'misaligned': ':'}

# ================== 运行试验并记录数据 ==================
raw_data = []   # 存储每次试验的详细信息

for case in cases:
    gd_iters = []   # 存储收敛迭代次数
    nes_iters = []
    for seed in range(n_avg):
        np.random.seed(seed)
        Sigma0 = generate_Sigma0(case, eig_H)
        x0 = np.random.multivariate_normal(np.zeros(d), Sigma0)
        
        # GD
        fvals_gd = gradient_descent(x0, grad, L_max, T_max, f)
        idx_gd = np.where(fvals_gd <= threshold)[0]
        iter_gd = idx_gd[0] if len(idx_gd) > 0 else T_max
        gd_iters.append(iter_gd)
        raw_data.append(['GD', case, seed, iter_gd, 1 if iter_gd < T_max else 0])
        
        # Nesterov
        fvals_nes = nesterov(x0, grad, L_max, T_max, f)
        idx_nes = np.where(fvals_nes <= threshold)[0]
        iter_nes = idx_nes[0] if len(idx_nes) > 0 else T_max
        nes_iters.append(iter_nes)
        raw_data.append(['Nesterov', case, seed, iter_nes, 1 if iter_nes < T_max else 0])

# 保存原始数据
df_raw = pd.DataFrame(raw_data, columns=['algorithm', 'case', 'seed', 'iteration', 'success'])
df_raw.to_csv('four_cases_raw.csv', index=False)

# ================== 统计汇总 ==================
def case_stats(df, algo, case):
    sub = df[(df['algorithm'] == algo) & (df['case'] == case)]
    succ = sub[sub['success'] == 1]
    succ_rate = len(succ) / len(sub) * 100
    if len(succ) > 0:
        iters = succ['iteration'].values
        median = np.median(iters)
        try:
            res = bootstrap((iters,), np.median, confidence_level=0.95,
                            method='BCa', random_state=42)
            ci_low, ci_high = res.confidence_interval
        except:
            ci_low, ci_high = np.nan, np.nan
    else:
        median = ci_low = ci_high = np.nan
    return succ_rate, median, ci_low, ci_high

algorithms = ['GD', 'Nesterov']
summary_rows = []
for algo in algorithms:
    for case in cases:
        rate, med, lo, hi = case_stats(df_raw, algo, case)
        summary_rows.append([algo, case, rate, med, lo, hi])

df_summary = pd.DataFrame(summary_rows, columns=['algorithm', 'case', 'success_rate(%)', 'median', 'ci_low', 'ci_high'])
df_summary.to_csv('four_cases_stats.csv', index=False)

print("\n统计汇总 (已保存至 four_cases_stats.csv):")
print(df_summary.to_string())

# ================== 绘图（图1） ==================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'lines.linewidth': 1.2
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True)

# 重新计算平均曲线（只为了绘图，可用已存数据简化，这里直接重新运行小样本平均）
n_avg_plot = 20   # 绘图用平均曲线，取前20次即可
for case in cases:
    gd_curves, nes_curves = [], []
    for seed in range(n_avg_plot):
        np.random.seed(seed)
        Sigma0 = generate_Sigma0(case, eig_H)
        x0 = np.random.multivariate_normal(np.zeros(d), Sigma0)
        gd_curves.append(gradient_descent(x0, grad, L_max, T_max, f))
        nes_curves.append(nesterov(x0, grad, L_max, T_max, f))
    # 截断到相同长度
    min_len_gd = min(len(c) for c in gd_curves)
    avg_gd = np.mean([c[:min_len_gd] for c in gd_curves], axis=0)
    min_len_nes = min(len(c) for c in nes_curves)
    avg_nes = np.mean([c[:min_len_nes] for c in nes_curves], axis=0)
    
    ax1.semilogy(avg_gd, color=colors[case], linestyle=linestyles[case], label=case_names[cases.index(case)])
    ax2.semilogy(avg_nes, color=colors[case], linestyle=linestyles[case], label=case_names[cases.index(case)])

ax1.set_ylabel('Optimality gap')
ax1.set_title('Gradient Descent')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='best')
ax1.axhline(y=threshold, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Optimality gap')
ax2.set_title('Nesterov')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='best')
ax2.axhline(y=threshold, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

ax1.set_xlim(0, 1500)
ax2.set_xlim(0, 1500)
plt.tight_layout()
plt.savefig('gd_nesterov_four_cases.pdf')
print("收敛曲线图已保存为 gd_nesterov_four_cases.pdf")
plt.show()

# ================== 统计检验（与 isotropic 比较） ==================
print("\n统计检验结果 (Mann-Whitney U vs isotropic):")
for algo in algorithms:
    for case in ['aligned', 'misaligned', 'independent']:
        sub_case = df_raw[(df_raw['algorithm'] == algo) & (df_raw['case'] == case) & (df_raw['success'] == 1)]['iteration'].values
        sub_iso = df_raw[(df_raw['algorithm'] == algo) & (df_raw['case'] == 'isotropic') & (df_raw['success'] == 1)]['iteration'].values
        if len(sub_case) > 0 and len(sub_iso) > 0:
            u, p = mannwhitneyu(sub_case, sub_iso, alternative='two-sided')
            print(f"{algo:8s} {case:12s}: p = {p:.4f}")