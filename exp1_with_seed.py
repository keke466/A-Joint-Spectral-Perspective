# -*- coding: utf-8 -*-
"""
实验一：四种初始化类型对比（GD 与 Nesterov）
接受一个种子参数，运行后将统计结果保存到 CSV 文件。
使用方法：python exp1_with_seed.py <seed>
"""

import numpy as np
import pandas as pd
import sys
from scipy.stats import beta, bootstrap

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

# ================== 获取种子 ==================
if len(sys.argv) > 1:
    master_seed = int(sys.argv[1])
else:
    master_seed = 42   # 默认种子

print(f"Running with master seed = {master_seed}")

np.random.seed(master_seed)

# ================== 实验参数 ==================
d = 50
xi, tau = -0.5, 0.5
L = 10.0
T_max = 2000
n_avg = 50                     # 试验次数
threshold = 1e-6

# 生成 Hessian 特征值（固定，但依赖种子）
eig_H = beta.rvs(a=xi+1, b=tau+1, size=d) * L
H = np.diag(eig_H)
L_max = np.max(eig_H)

def f(x): return 0.5 * x @ H @ x
def grad(x): return H @ x

cases = ['isotropic', 'independent', 'aligned', 'misaligned']

# ================== 运行试验并记录数据 ==================
raw_data = []   # 存储每次试验的详细信息

for case in cases:
    gd_iters = []   # 存储收敛迭代次数
    nes_iters = []
    for seed in range(n_avg):
        # 注意：这里用 seed 作为子随机种子，但整个实验已经由 master_seed 控制了整体随机性
        # 为了确保每个试验内部独立，我们在每次循环中重置随机状态？不，我们直接用 np.random 会累积，但结果仍由 master_seed 确定。
        # 为了更清晰，我们可以在循环内使用独立的随机数生成器，但当前方式也 OK。
        # 这里保持与之前代码一致，用 np.random 连续采样。
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
                            method='BCa', random_state=master_seed)
            ci_low, ci_high = res.confidence_interval
        except:
            ci_low, ci_high = np.nan, np.nan
    else:
        median = ci_low = ci_high = np.nan
    return succ_rate, median, ci_low, ci_high

df_raw = pd.DataFrame(raw_data, columns=['algorithm', 'case', 'seed', 'iteration', 'success'])

algorithms = ['GD', 'Nesterov']
summary_rows = []
for algo in algorithms:
    for case in cases:
        rate, med, lo, hi = case_stats(df_raw, algo, case)
        summary_rows.append([algo, case, rate, med, lo, hi])

df_summary = pd.DataFrame(summary_rows, columns=['algorithm', 'case', 'success_rate(%)', 'median', 'ci_low', 'ci_high'])

# 保存结果到文件，文件名包含种子
output_file = f'four_cases_stats_seed_{master_seed}.csv'
df_summary.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# 可选：也保存原始数据，但为了节省空间，只保存汇总
# df_raw.to_csv(f'four_cases_raw_seed_{master_seed}.csv', index=False)