# collect_results.py
import pandas as pd
import glob

# 你实际运行过的种子列表（如果还有别的种子，请添加或删除）
seeds = [42, 100, 2024, 5678, 9999]
records = []

for seed in seeds:
    filename = f'four_cases_stats_seed_{seed}.csv'
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"警告：文件 {filename} 不存在，跳过该种子。")
        continue

    # 只提取 Nesterov 的结果（如果你也需要 GD，可以去掉过滤）
    df_nes = df[df['algorithm'] == 'Nesterov'].copy()

    for _, row in df_nes.iterrows():
        records.append({
            'seed': seed,
            'case': row['case'],
            'success_rate': row['success_rate(%)'],   # 添加成功率
            'median': row['median'],
            'ci_low': row['ci_low'],
            'ci_high': row['ci_high']
        })

if not records:
    print("没有找到任何数据，请检查种子列表和CSV文件是否存在。")
else:
    df_all = pd.DataFrame(records)

    # 保存详细记录（包含成功率和置信区间）
    df_all.to_csv('sensitivity_analysis_detailed.csv', index=False, encoding='utf-8-sig')
    print("\n详细记录已保存到 sensitivity_analysis_detailed.csv")

    # 生成中位数透视表（与之前一样）
    pivot_median = df_all.pivot(index='seed', columns='case', values='median')
    print("\n各种子下 Nesterov 的中位数迭代次数：")
    print(pivot_median)

    # 可选：生成成功率透视表
    pivot_success = df_all.pivot(index='seed', columns='case', values='success_rate')
    print("\n各种子下 Nesterov 的成功率（%）：")
    print(pivot_success)

    # 计算 aligned 相对于 isotropic 的加速百分比
    pivot_median['speedup_%'] = (pivot_median['isotropic'] - pivot_median['aligned']) / pivot_median['isotropic'] * 100
    print("\n加速百分比 (aligned vs isotropic):")
    print(pivot_median['speedup_%'])

    # 保存透视表（包含中位数和加速百分比）
    pivot_median.to_csv('sensitivity_analysis_nesterov.csv', encoding='utf-8-sig')
    print("\n汇总表已保存到 sensitivity_analysis_nesterov.csv")