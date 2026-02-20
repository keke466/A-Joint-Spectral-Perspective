# run_multiple_seeds.py
import subprocess
import sys

seeds = [42, 100, 2024, 5678, 9999]   # 你可以增加或修改种子列表

for seed in seeds:
    print(f"\n=== Running with seed {seed} ===")
    # 调用实验脚本，等待其完成
    result = subprocess.run([sys.executable, 'exp1_with_seed.py', str(seed)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    print(f"Finished seed {seed}")