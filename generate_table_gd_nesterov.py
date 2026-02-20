# 文件名: generate_table_gd_nesterov.py
import numpy as np
from scipy.stats import beta, mannwhitneyu

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

# 实验参数
d = 50
xi, tau = -0.5, 0.5
L = 10.0
T_max = 2000
n_trials = 20
threshold = 1e-6

np.random.seed(42)
eig_H = beta.rvs(a=xi+1, b=tau+1, size=d) * L
H = np.diag(eig_H)
L_max = np.max(eig_H)

def f(x): return 0.5 * x @ H @ x
def grad(x): return H @ x

cases = ['isotropic', 'independent', 'aligned', 'misaligned']
results = {case: {'gd': [], 'nesterov': []} for case in cases}

for trial in range(n_trials):
    print(f"Trial {trial+1}/{n_trials}")
    for case in cases:
        Sigma0 = generate_Sigma0(case, eig_H)
        x0 = np.random.multivariate_normal(np.zeros(d), Sigma0)
        
        # GD
        fvals_gd = gradient_descent(x0, grad, L_max, T_max, f)
        idx_gd = np.where(fvals_gd <= threshold)[0]
        results[case]['gd'].append(idx_gd[0] if len(idx_gd) > 0 else np.inf)
        
        # Nesterov
        fvals_nes = nesterov(x0, grad, L_max, T_max, f)
        idx_nes = np.where(fvals_nes <= threshold)[0]
        results[case]['nesterov'].append(idx_nes[0] if len(idx_nes) > 0 else np.inf)

# 统计函数
def bootstrap_ci(data, n_bootstrap=10000):
    data = np.array(data)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return [np.nan, np.nan]
    medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        medians.append(np.median(sample))
    return np.percentile(medians, [2.5, 97.5])

def format_result(case, algo):
    data = results[case][algo]
    success = [x for x in data if x < np.inf]
    n_success = len(success)
    if n_success == 0:
        return f"$>${T_max} (0/{n_trials})"
    median = np.median(success)
    ci = bootstrap_ci(success)
    if np.isnan(ci[0]):
        ci_str = ""
    else:
        ci_str = f" [{ci[0]:.0f},{ci[1]:.0f}]"
    return f"{median:.0f}{ci_str} ({n_success}/{n_trials})"

# 输出 LaTeX 表格
print("\\begin{table*}[htbp]")
print("    \\centering")
print("    \\caption{Effect of initialization structure on convergence (2000 iterations, threshold $10^{-6}$). Medians and 95\\% confidence intervals are computed from successful trials; numbers in parentheses indicate success rates.}")
print("    \\label{tab:initialization_impact}")
print("    \\begin{tabular}{lcc}")
print("        \\toprule")
print("        Initialization type & GD & Nesterov \\\\")
print("        \\midrule")
for case in cases:
    case_display = case.capitalize()
    if case == 'independent':
        case_display = 'Independent anisotropic'
    elif case == 'aligned':
        case_display = 'Aligned anisotropic (good)'
    elif case == 'misaligned':
        case_display = 'Misaligned anisotropic (bad)'
    gd_str = format_result(case, 'gd')
    nes_str = format_result(case, 'nesterov')
    print(f"        {case_display} & {gd_str} & {nes_str} \\\\")
print("        \\bottomrule")
print("    \\end{tabular}")
print("\\end{table*}")