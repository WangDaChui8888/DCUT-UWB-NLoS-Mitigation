import subprocess
from pathlib import Path
import time, os, shutil

# -------- 配置 --------
NUM_RUNS = 10
BASE_SEED = 42
MAIN_SCRIPT = "main_script.py"   # 你的主脚本文件名
PROJECT_DIR = Path(__file__).resolve().parent
BASE_SAVE_DIR = PROJECT_DIR / "Ghent_Statistical_Runs"
# 预测时 GPU 偶发崩溃的常见退出码
SEGV_CODES = {-11, -6}

def _base_env():
    """给每个子进程一套稳定的环境变量（限制底层线程，静音 TF 啰嗦日志）。"""
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # 限制 BLAS/NumExpr 线程，降低崩溃概率
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env.setdefault(k, "1")
    return env

def _cpu_env():
    """强制 CPU 环境（mac 上也适用：禁用 Metal/MPS）。"""
    env = _base_env()
    env["CUDA_VISIBLE_DEVICES"] = "-1"   # Linux/Windows 生效
    env["TF_MPS_ENABLED"] = "0"          # macOS 关闭 Metal
    return env

if __name__ == "__main__":
    start_time = time.time()
    BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Starting {NUM_RUNS} runs in: {PROJECT_DIR}")

    successes, failures = 0, 0

    for i in range(NUM_RUNS):
        seed = BASE_SEED + i
        save_dir = BASE_SAVE_DIR / f"run_seed_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*90)
        print(f"🎬 Run {i+1}/{NUM_RUNS} | SEED={seed} | Out: {save_dir}")
        print("="*90)

        cmd = ["python", MAIN_SCRIPT, "--seed", str(seed), "--save_dir", str(save_dir)]

        # 1) 先试 GPU（或默认设备）
        result = subprocess.run(cmd, cwd=PROJECT_DIR, env=_base_env())
        rc = result.returncode

        if rc in SEGV_CODES:
            print(f"⚠️  Run(seed={seed}) crashed with rc={rc} (likely GPU). Retrying on CPU...")
            # 清理可能的半成品目录，避免后续解析混乱
            try:
                for p in save_dir.glob("*"):
                    if p.is_file(): p.unlink()
            except Exception:
                pass
            # 2) CPU 兜底再试一次
            result = subprocess.run(cmd, cwd=PROJECT_DIR, env=_cpu_env())
            rc = result.returncode

        if rc == 0:
            print(f"✅ SUCCESS: seed {seed}")
            successes += 1
        else:
            print(f"❌ FAIL: seed {seed} (rc={rc}) — see logs above.")
            failures += 1
            # 不中断，继续跑其他随机种子
            continue

    dur = (time.time() - start_time)/60
    print("\n" + "*"*90)
    print(f"🎉 Finished runs in {dur:.2f} min.  Success: {successes} / {NUM_RUNS} | Failures: {failures}")
    print(f"📂 Results root: {BASE_SAVE_DIR}")
    print("👉 Next: run 'analyze_results.py' to compute means, 95% CIs, and p-values.")
    print("*"*90)
