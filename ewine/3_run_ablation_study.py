# -*- coding: utf-8 -*-
"""
Ablation Runner & Analyzer (v2.8 - 已分离绘图功能)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# 可选库
# Matplotlib 仅用于检查是否存在，但不再用于此脚本中的绘图
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
try:
    from scipy.stats import wilcoxon
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ============================ 基本配置 ============================
DEFAULT_MAIN_SCRIPT = "main_script.py"
DEFAULT_MODEL_NAME  = "DualTransformer"

BASELINE_PARAMS = {
    '--transformer_layers': '6',
    '--attention_heads':    '8',
    '--loss_weight_error':  '0.6',
    '--learning_rate':      '0.0002',
    '--dropout_rate':       '0.1',
    '--ff_dim':             '256',
    '--lambda_mask':        '1.0',
    '--lambda_ga':          '1.0',
}
GRID_LAYERS = [2, 4, 6, 8]
GRID_HEADS  = [2, 4, 8]
GRID_LOSSW  = [0.3, 0.6, 0.9]
GRID_LM = [0.0, 0.5, 1.0, 2.0]
GRID_LG = [0.0, 0.5, 1.0, 2.0]
DEFAULT_SEEDS = [17, 23, 42]

PROJECT_DIR   = Path(__file__).resolve().parent
BASE_SAVE_DIR = Path(os.environ.get("ABLATION_SAVE_DIR", str(PROJECT_DIR / "Ablation_Study_Runs_Advanced")))
MASTER_LOG_PATH = Path(os.environ.get("ABLATION_MASTER_LOG", str(PROJECT_DIR / "ablation_master_log.txt")))

# ============================ 工具函数 ============================
def parse_num_list(s: str, typ=float) -> List[int|float]:
    """将逗号分隔的字符串解析为数字列表"""
    if not s: return []
    return [typ(x.strip()) for x in s.split(",") if x.strip()]

def ensure_dir(p: Path) -> Path:
    """确保目录存在"""
    p.mkdir(parents=True, exist_ok=True)
    return p

def build_params_list(d: Dict[str, str]) -> List[str]:
    """将参数字典转换为命令行列表"""
    out = []
    for k, v in d.items():
        out.extend([k, str(v)])
    return out

def _fsync_write(fh, text: str):
    """带强制刷新的文件写入，确保日志实时性"""
    fh.write(text)
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except Exception:
        pass

def _ensure_master_open_banner(fh):
    """在主日志文件开头写入一个时间戳横幅"""
    _fsync_write(fh, "\n--- Ablation Started @ %s ---\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def get_summary_filename() -> str:
    """获取标准的摘要文件名"""
    try:
        from config import SUMMARY_CSV_FILENAME
        return SUMMARY_CSV_FILENAME
    except Exception:
        return "summary.csv"

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """将DataFrame的列名标准化（小写，下划线）"""
    df = df.copy()
    df.columns = [c.replace(" ", "_").replace("-", "_").lower() for c in df.columns]
    return df

def _pick_row(df: pd.DataFrame, prefer_name: str = DEFAULT_MODEL_NAME) -> pd.Series:
    """从摘要DataFrame中挑选出目标模型的行"""
    try:
        idx_lower = [str(i).lower() for i in df.index]
        if prefer_name.lower() in idx_lower:
            return df.iloc[idx_lower.index(prefer_name.lower())]
    except Exception:
        pass
    return df.iloc[0]

def extract_metrics_from_summary(summary_csv: Path) -> Dict[str, float]:
    """从单个运行的summary.csv中提取关键指标"""
    out: Dict[str, float] = {}
    if not summary_csv.exists():
        return out
    try:
        df = pd.read_csv(summary_csv, index_col=0)
        df = _normalize_cols(df)
        row = _pick_row(df)
        for key in ("balanced_accuracy_cv_mean", "rmse_mm_cv_mean"):
            if key in df.columns:
                val = row[key] if key in row.index else df[key].iloc[0]
                out[key] = float(val)
    except Exception:
        pass
    return out

# 轻量行解析（可附加到stdout事件里，便于检索）
_RE_EPOCH = re.compile(r'^\s*Epoch\s+(\d+)\s*/\s*(\d+)\s*')
_RE_FOLD  = re.compile(r'Processing Fold\s+(\d+)\s*/\s*(\d+)')
_RE_STEP  = re.compile(r'\[Step\s+(\d+)\]\s*([^-]+?)(?:\s*-{3,}|$)')
_RE_KEYS  = re.compile(r'([A-Za-z0-9_]+)\s*:\s*([0-9.eE+\-]+)')

def parse_line(line: str) -> Dict:
    """解析单行日志，提取结构化信息"""
    info = {}
    s = line.strip()
    if not s:
        return info
    m = _RE_EPOCH.search(s)
    if m:
        info["epoch"] = int(m.group(1))
        info["epochs_total"] = int(m.group(2))
    m = _RE_FOLD.search(s)
    if m:
        info["fold"] = int(m.group(1))
        info["folds_total"] = int(m.group(2))
    m = _RE_STEP.search(s)
    if m:
        info["step_id"] = int(m.group(1))
        info["step_name"] = m.group(2).strip()
    pairs = dict((k.replace("-", "_"), v) for k, v in _RE_KEYS.findall(s))
    for k, v in list(pairs.items()):
        try:
            pairs[k] = float(v)
        except Exception:
            pass
    if pairs:
        info["kv"] = pairs
    if "error" in s.lower():
        info["level"] = "error"
    elif "warning" in s.lower() or "warn" in s.lower():
        info["level"] = "warning"
    elif "info" in s.lower():
        info["level"] = "info"
    return info

# ============================ 生成实验列表 ============================
def gen_baseline_run(baseline_params: Dict[str, str]) -> List[Dict]:
    """生成基线模型的运行配置"""
    return [{'name': "Baseline_Full_Model", 'params': baseline_params.copy()}]

def gen_one_factor_runs(baseline_params: Dict[str, str], layers, heads, lossw) -> List[Dict]:
    """生成单因素变量实验的配置列表"""
    runs = []
    for L in layers:
        if L == int(baseline_params['--transformer_layers']): continue
        p = baseline_params.copy(); p['--transformer_layers'] = str(L)
        runs.append({'name': f"Layers_{L}", 'params': p})
    for H in heads:
        if H == int(baseline_params['--attention_heads']): continue
        p = baseline_params.copy(); p['--attention_heads'] = str(H)
        runs.append({'name': f"Heads_{H}", 'params': p})
    for W in lossw:
        if abs(W - float(baseline_params['--loss_weight_error'])) < 1e-12: continue
        p = baseline_params.copy(); p['--loss_weight_error'] = str(W)
        runs.append({'name': f"LossWeight_{W:.1f}", 'params': p})
    return runs

def gen_reg_strength_grid(baseline_params: Dict[str, str], lms, lgs) -> List[Dict]:
    """生成正则化强度网格搜索的配置列表"""
    runs = []
    for lm in lms:
        for lg in lgs:
            if (abs(lm - float(baseline_params['--lambda_mask'])) < 1e-12 and
                abs(lg - float(baseline_params['--lambda_ga']))   < 1e-12):
                continue
            p = baseline_params.copy()
            p['--lambda_mask'] = str(lm)
            p['--lambda_ga']   = str(lg)
            runs.append({'name': f"RegGrid_mask{lm:.1f}_ga{lg:.1f}", 'params': p})
    return runs

# ============================ 单次运行（实时写主日志） ============================
def _build_cmd_and_paths(py_exec: str, main_script: str, model_name: str,
                         run_name: str, seed: int, params_dict: Dict[str, str]):
    """构建单次运行的命令行和保存路径"""
    save_dir = BASE_SAVE_DIR / model_name / run_name / f"seed_{seed}"
    ensure_dir(save_dir)
    params_list = build_params_list(params_dict)
    cmd = [py_exec, main_script, "--seed", str(seed),
           "--save_dir", str(save_dir), "--run_only", model_name] + params_list
    return cmd, save_dir

def run_single(main_script: str, model_name: str, run_cfg: Dict, seed: int,
               cwd: Path, py_exec: str, mastlog_fh) -> Tuple[bool, Path, Dict]:
    """执行单次实验，并实时记录日志"""
    run_name, params_dict = run_cfg['name'], run_cfg['params']
    cmd, save_dir = _build_cmd_and_paths(py_exec, main_script, model_name, run_name, seed, params_dict)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr = f"[{now}] Running: {run_name} | seed={seed}"
    print("\n" + "="*len(hdr)); print(hdr); print(f"  CMD: {' '.join(cmd)}"); print("="*len(hdr))

    t0 = time.time()
    start_iso = datetime.now().isoformat()
    run_log = save_dir / "run.log"
    stdout_jsonl = save_dir / "stdout.jsonl"

    with open(run_log, "w", encoding="utf-8") as logf, open(stdout_jsonl, "a", encoding="utf-8") as outjson:
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding='utf-8', errors='replace'
            )
            for raw in iter(proc.stdout.readline, ''):
                print(raw, end=''); logf.write(raw)
                rec = {
                    "event": "stdout", "ts": time.time(), "name": run_name, "seed": seed,
                    "elapsed": round(time.time() - t0, 3), "line": raw.rstrip("\n")
                }
                parsed = parse_line(raw)
                if parsed: rec.update(parsed)
                _fsync_write(outjson, json.dumps(rec, ensure_ascii=False) + "\n")
                _fsync_write(mastlog_fh, json.dumps(rec, ensure_ascii=False) + "\n")

            proc.wait()
            ok = (proc.returncode == 0)
            elapsed = time.time() - t0
            end_iso = datetime.now().isoformat()
            summary_csv = save_dir / get_summary_filename()
            metrics = extract_metrics_from_summary(summary_csv)
            record = {
                "ts": time.time(), "name": run_name, "seed": seed, "ok": ok, "cmd": " ".join(cmd),
                "params": params_dict, "save_dir": str(save_dir), "run_log": str(run_log),
                "stdout_jsonl": str(stdout_jsonl), "execution_main_log": str(save_dir / "execution_main.log"),
                "summary_csv": str(summary_csv), "started_at": start_iso, "finished_at": end_iso,
                "elapsed_sec": round(elapsed, 3), **metrics
            }
            with open(save_dir / "ablation_record.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            if ok:
                print(f"[OK] {run_name} seed={seed}\n")
            else:
                msg = f"[FAIL] {run_name} seed={seed} exit={proc.returncode}\n"
                print(msg); logf.write(f"\n{msg}\n")

            return ok, save_dir, record

        except Exception as e:
            elapsed = time.time() - t0
            end_iso = datetime.now().isoformat()
            record = {
                "ts": time.time(), "name": run_name, "seed": seed, "ok": False, "cmd": " ".join(cmd),
                "params": params_dict, "save_dir": str(save_dir), "run_log": str(run_log),
                "stdout_jsonl": str(stdout_jsonl), "execution_main_log": str(save_dir / "execution_main.log"),
                "summary_csv": str(save_dir / get_summary_filename()), "started_at": start_iso,
                "finished_at": end_iso, "elapsed_sec": round(elapsed, 3), "exception": str(e)
            }
            with open(save_dir / "ablation_record.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            exc_line = {
                "event": "stdout", "ts": time.time(), "name": run_name, "seed": seed,
                "elapsed": round(time.time() - t0, 3), "level": "error", "line": f"[EXCEPTION] {e}"
            }
            with open(stdout_jsonl, "a", encoding="utf-8") as outjson2:
                _fsync_write(outjson2, json.dumps(exc_line, ensure_ascii=False) + "\n")
            _fsync_write(mastlog_fh, json.dumps(exc_line, ensure_ascii=False) + "\n")

            msg = f"[EXC] {run_name} seed={seed} exception: {e}"
            print(msg); logf.write(f"\n{msg}\n{traceback.format_exc()}\n")
            return False, save_dir, record

# ============================ 批量运行 ============================
def run_all(main_script: str, model_name: str, runlist: List[Dict],
            seeds: List[int], cwd: Path, py_exec: str):
    """批量执行所有实验配置"""
    t0 = time.time()
    ensure_dir(BASE_SAVE_DIR); ensure_dir(MASTER_LOG_PATH.parent)
    print(f"[MasterLog] writing to: {MASTER_LOG_PATH}")

    with open(MASTER_LOG_PATH, "a", encoding="utf-8") as mastlog:
        _ensure_master_open_banner(mastlog)
        for cfg in runlist:
            for sd in seeds:
                cmd_preview, save_dir_preview = _build_cmd_and_paths(py_exec, main_script, model_name, cfg['name'], sd, cfg['params'])
                start_rec = {
                    "event": "run_start", "ts": time.time(), "name": cfg['name'], "seed": sd,
                    "cmd": " ".join(cmd_preview), "save_dir": str(save_dir_preview), "params": cfg['params']
                }
                _fsync_write(mastlog, json.dumps(start_rec, ensure_ascii=False) + "\n")
                ok, save_dir, record = run_single(main_script, model_name, cfg, sd, cwd=cwd, py_exec=py_exec, mastlog_fh=mastlog)

                rec_path = save_dir / "ablation_record.json"
                if rec_path.exists():
                    try:
                        with open(rec_path, "r", encoding="utf-8") as f:
                            end_rec = {"event": "run_end", **json.load(f)}
                        _fsync_write(mastlog, json.dumps(end_rec, ensure_ascii=False) + "\n")
                        continue
                    except Exception: pass
                
                end_rec_fallback = {
                    "event": "run_end", "ts": time.time(), "name": cfg['name'], "seed": sd, "ok": ok,
                    "save_dir": str(save_dir), "run_log": str(save_dir / "run.log"), "stdout_jsonl": str(save_dir / "stdout.jsonl"),
                    "execution_main_log": str(save_dir / "execution_main.log"), "summary_csv": str(save_dir / get_summary_filename()),
                    "params": cfg["params"],
                }
                _fsync_write(mastlog, json.dumps(end_rec_fallback, ensure_ascii=False) + "\n")

    print(f"[DONE] All runs finished in {(time.time()-t0)/60.0:.2f} min.")
    print(f"[MasterLog] updated -> {MASTER_LOG_PATH}")

# ============================ 汇总分析 ============================
def aggregate_runs(model_name: str, runlist: List[Dict], seeds: List[int]):
    """汇总所有运行结果，并生成最终的摘要表格"""
    print("\n" + "*"*80); print("📊 Analyzing all ablation study results..."); print("*"*80)
    summary_filename = get_summary_filename()
    metrics = ['balanced_accuracy_cv_mean', 'rmse_mm_cv_mean']
    rows = []

    def load_vec(run_name: str, col: str) -> np.ndarray:
        vals = []
        for sd in seeds:
            f = BASE_SAVE_DIR / model_name / run_name / f"seed_{sd}" / summary_filename
            if not f.exists(): continue
            try:
                df = pd.read_csv(f, index_col=0)
                df = _normalize_cols(df)
                if col in df.columns:
                    vals.append(_pick_row(df)[col])
            except Exception: pass
        return pd.to_numeric(pd.Series(vals), errors='coerce').dropna().values

    base_vecs = {m: load_vec("Baseline_Full_Model", m) for m in metrics}

    for cfg in runlist:
        name = cfg['name']
        row = {'Configuration': name}
        for m in metrics:
            v = load_vec(name, m)
            if v.size > 0:
                row[f"{m}_mean"] = float(np.mean(v))
                row[f"{m}_std"]  = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
                base = base_vecs.get(m, np.array([]))
                if base.size > 0:
                    row[f"Delta_{m}"] = float(np.mean(v) - np.mean(base))
                    if _HAS_SCIPY and v.size == base.size and v.size >= 5:
                        try:
                            row[f"Pval_{m}"] = float(wilcoxon(v, base).pvalue)
                        except Exception: pass
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Configuration")
    out_csv = BASE_SAVE_DIR / "ablation_summary_table.csv"
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv)
    print(df.to_string(float_format="%.4f"))
    print(f"\n[OK] Summary table saved -> {out_csv}")

    # ===== 热力图绘制代码已被移除 =====
    # 现在此脚本只负责生成上面的CSV文件。
    # 请使用独立的 plot_heatmap.py 脚本来生成热力图。
    print("\n[INFO] Heatmap generation has been moved to a separate script.")
    print(f"[INFO] To generate the heatmap, run: python plot_heatmap.py --input {out_csv}")


# ============================ 入口 ============================
def main():
    ap = argparse.ArgumentParser(description="Extensive Ablation Runner & Analyzer")
    ap.add_argument("--run", action="store_true", help="Run ablation experiments")
    ap.add_argument("--analyze", action="store_true", help="Aggregate results and produce summary CSV")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)), help="Comma-separated seeds")
    ap.add_argument("--no-one-factor", action="store_true", help="Disable one-factor ablation")
    ap.add_argument("--no-reg-grid", action="store_true", help="Disable regularizer grid scan")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to launch the main script")
    ap.add_argument("--main", type=str, default=DEFAULT_MAIN_SCRIPT, help="Main training script filename")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name for --run_only")
    args = ap.parse_args()

    seeds = parse_num_list(args.seeds, int)
    runlist: List[Dict] = []
    runlist.extend(gen_baseline_run(BASELINE_PARAMS))
    if not args.no_one-factor:
        runlist.extend(gen_one_factor_runs(BASELINE_PARAMS, GRID_LAYERS, GRID_HEADS, GRID_LOSSW))
    if not args.no-reg-grid:
        runlist.extend(gen_reg_strength_grid(BASELINE_PARAMS, GRID_LM, GRID_LG))

    if args.run:
        run_all(args.main, args.model, runlist, seeds, cwd=PROJECT_DIR, py_exec=args.python)
    if args.analyze:
        aggregate_runs(args.model, runlist, seeds)
    if not args.run and not args.analyze:
        print("[Info] Nothing to do. Use --run and/or --analyze. See --help.")

if __name__ == "__main__":
    main()