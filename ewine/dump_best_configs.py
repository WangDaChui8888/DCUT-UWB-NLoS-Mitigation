
# -*- coding: utf-8 -*-
import json, re, csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "Ghent_Statistical_Runs" / "ba_best_runs.csv"
TUNE_ROOT = ROOT / "Tuning_Baselines_R1"

# TUNING GRID (keys without leading dashes for output columns)
TUNING_GRID = {
    "ST": ["transformer_layers","attention_heads","ff_dim","loss_weight_error","learning_rate","dropout_rate"],
    "DNN": ["dense_units","dropout_rate","learning_rate","weight_decay","activation","loss_weight_error"],
    "LS-SVM": ["svm_kernel","svm_C","svm_gamma","svm_class_weight","standardize"],
    "XGBoost": ["xgb_n_estimators","xgb_max_depth","xgb_learning_rate","xgb_subsample","xgb_colsample_bytree","xgb_gamma","xgb_min_child_weight"],
    "CNN-LSTM": ["conv_filters","kernel_size","pool_size","lstm_units","dense_units","dropout_rate","learning_rate","activation","loss_weight_error"],
}

NAME_MAP = {"ST":"SingleTransformer","CNN-LSTM":"CNNLSTM", "DNN":"DNN", "XGBoost":"XGBoost", "LS-SVM":"LS-SVM"}

CONFIG_FILES_JSON = ["config.json","hparams.json","args.json","params.json"]
CONFIG_FILES_YAML = ["config.yaml","hparams.yaml","params.yaml"]
LOG_FILES = ["train.log","stdout.log","stderr.log","run.log"]

def read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            return None

def read_yaml(p: Path) -> Optional[dict]:
    try:
        import yaml
    except Exception:
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def parse_cli_in_text(s: str) -> Dict[str,str]:
    # Parse patterns like "--key value" or "key=value"
    out = {}
    # --key value
    for m in re.finditer(r"(?:^|\s)--([A-Za-z0-9_]+)\s+([^\s]+)", s):
        out[m.group(1)] = m.group(2)
    # key=value
    for m in re.finditer(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s,]+)", s):
        out[m.group(1)] = m.group(2)
    return out

def most_common_str(values: List[str]) -> str:
    values = [v for v in values if isinstance(v,str) and v.strip()]
    if not values: return ""
    counts = {}
    for v in values:
        s = v.strip()
        try:
            obj = json.loads(s)
            s = json.dumps(obj, sort_keys=True)
        except Exception:
            pass
        counts[s] = counts.get(s,0)+1
    return max(counts.items(), key=lambda kv: kv[1])[0]

def flatten_cfg(method: str, raw: Any) -> Dict[str, str]:
    keys = TUNING_GRID.get(method, [])
    kv = {}
    # raw could be dict / list / string
    if isinstance(raw, dict):
        for k in keys:
            for cand in [k, f"--{k}"]:
                if cand in raw:
                    kv[k] = str(raw[cand])
                    break
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                for k in keys:
                    for cand in [k, f"--{k}"]:
                        if cand in item and k not in kv:
                            kv[k] = str(item[cand]); break
    elif isinstance(raw, str):
        # try json
        try:
            obj = json.loads(raw)
            return flatten_cfg(method, obj)
        except Exception:
            pass
        # parse CLI
        parsed = parse_cli_in_text(raw)
        for k in keys:
            if k in parsed: kv[k] = parsed[k]
            elif f"--{k}" in parsed: kv[k] = parsed[f"--{k}"]
    return kv

def recover_cfg_from_run(run_dir: Path) -> Dict[str,str]:
    # 1) try in the CSV (best_cfg column)
    csv_path = run_dir / "all_models_summary_combined_3fold_cv_v7.csv"
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if "best_cfg" in df.columns:
                raw = most_common_str([str(x) for x in df["best_cfg"].tolist()])
                return {"__raw__": raw}
        except Exception:
            pass
    # 2) try JSON files
    for fname in CONFIG_FILES_JSON:
        p = run_dir / fname
        if p.exists():
            data = read_json(p)
            if isinstance(data, dict) or isinstance(data, list):
                return {"__raw__": json.dumps(data)}
    # 3) try YAML files
    for fname in CONFIG_FILES_YAML:
        p = run_dir / fname
        if p.exists():
            data = read_yaml(p)
            if data is not None:
                try:
                    return {"__raw__": json.dumps(data)}
                except Exception:
                    return {"__raw__": str(data)}
    # 4) try logs
    blobs = []
    for fname in LOG_FILES + [p.name for p in run_dir.glob("*.txt")]:
        p = run_dir / fname
        if p.exists() and p.is_file():
            try:
                blobs.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    if blobs:
        merged = "\n".join(blobs)
        parsed = parse_cli_in_text(merged)
        if parsed:
            return {"__raw__": json.dumps(parsed)}
        return {"__raw__": merged[:2000]}
    return {"__raw__": ""}

def main():
    if not BEST.exists():
        raise FileNotFoundError(f"Missing {BEST}")
    df = pd.read_csv(BEST)
    rows = []
    for _, r in df.iterrows():
        method = str(r["Method"])
        algo_folder = NAME_MAP.get(method, method)
        run = str(r["best_run"])
        run_dir = TUNE_ROOT / algo_folder / run
        raw_cfg = recover_cfg_from_run(run_dir).get("__raw__", "")
        flat = flatten_cfg(method, raw_cfg)
        row = {"Method": method, "best_run": run, "raw_cfg": raw_cfg}
        for k in TUNING_GRID.get(method, []):
            row[k] = flat.get(k, "")
        rows.append(row)

    out_path = ROOT / "Ghent_Statistical_Runs" / "ba_best_configs_flat.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
