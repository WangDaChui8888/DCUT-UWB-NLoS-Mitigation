# -*- coding: utf-8 -*-
"""
SVM (SVC for Classification) and Ridge (Linear Regression) Experiment Module
Adapts experiment function for K-Fold cross-validation.
[FIX] Updated to use standardized scikit-learn parameter names from config.
"""
import numpy as np
import pandas as pd
import joblib
import time
import traceback
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

try:
    from evaluation_utils import evaluate_classification, evaluate_regression
    import config
except ImportError:
    print("[警告] svm_experiment.py: 无法导入 'evaluation_utils' 或 'config'。")
    def evaluate_classification(*args, **kwargs): return {"Accuracy": 0.0, "Balanced_Accuracy": 0.0, "Precision_NLOS": 0.0, "Recall_NLOS": 0.0, "F1_NLOS": 0.0, "TN": 0, "FP": 0, "FN": 0, "TP": 0}
    def evaluate_regression(*args, **kwargs): return {"RMSE_m": 0.0, "MAE_m": 0.0, "RMSE_mm": 0.0, "MAE_mm": 0.0}

def run_svm_ridge_fold_experiment(
    X_train_cir_scaled,
    y_train_nlos,
    y_train_error_scaled,
    X_val_cir_scaled,
    scaler_error,
    svm_params: dict,
    ridge_params: dict,
    fold_number: int,
    random_seed: int
    ):
    """
    在 K 折交叉验证的单个折叠上运行 SVC 和 Ridge 实验。
    """
    print(f"\n[SVC + Ridge 实验 - 折叠 {fold_number}] === 开始运行 ===")
    
    try:
        print(f"[SVC+Ridge 折叠 {fold_number}] 训练数据形状: CIR={X_train_cir_scaled.shape}, NLOS={y_train_nlos.shape}, Error={y_train_error_scaled.shape}")
        print(f"[SVC+Ridge 折叠 {fold_number}] 验证数据形状: CIR={X_val_cir_scaled.shape}")

        # --- 模型训练 ---
        print(f"[SVC+Ridge 折叠 {fold_number}] 训练 SVC (分类)...")
        svc_start_time = time.time()
        
        # --- 【关键修复】:直接使用 svm_params 字典，因为它现在包含标准名称 ---
        svc_model = SVC(**svm_params, random_state=random_seed)
        
        svc_model.fit(X_train_cir_scaled, y_train_nlos)
        svc_train_time = time.time() - svc_start_time
        print(f"[SVC+Ridge 折叠 {fold_number}] SVC 训练完成，耗时: {svc_train_time:.2f} 秒")

        print(f"[SVC+Ridge 折叠 {fold_number}] 训练 Ridge (回归)...")
        ridge_start_time = time.time()
        
        # --- 【关键修复】: 直接使用 ridge_params 字典 ---
        ridge_model = Ridge(**ridge_params, random_state=random_seed)
        
        ridge_model.fit(X_train_cir_scaled, y_train_error_scaled.flatten())
        ridge_train_time = time.time() - ridge_start_time
        print(f"[SVC+Ridge 折叠 {fold_number}] Ridge 训练完成，耗时: {ridge_train_time:.2f} 秒")
        
        train_times = {'svc': svc_train_time, 'ridge': ridge_train_time}

        # --- 在验证集上预测 ---
        print(f"[SVC+Ridge 折叠 {fold_number}] 在验证集上进行预测...")
        pred_start_time = time.time()

        y_pred_nlos_val = svc_model.predict(X_val_cir_scaled)
        
        if svm_params.get('probability', False):
            try:
                y_pred_nlos_prob_val = svc_model.predict_proba(X_val_cir_scaled)[:, 1]
            except (AttributeError, NotFittedError):
                y_pred_nlos_prob_val = np.full_like(y_pred_nlos_val, np.nan, dtype=float)
        else:
            y_pred_nlos_prob_val = np.full_like(y_pred_nlos_val, np.nan, dtype=float)

        y_pred_error_scaled_val = ridge_model.predict(X_val_cir_scaled)
        pred_time_val = time.time() - pred_start_time
        print(f"[SVC+Ridge 折叠 {fold_number}] 验证集预测完成，耗时: {pred_time_val:.4f} 秒")

        # --- 逆缩放验证集误差预测 ---
        print(f"[SVC+Ridge 折叠 {fold_number}] 逆缩放验证集预测误差...")
        y_pred_error_val_orig = scaler_error.inverse_transform(y_pred_error_scaled_val.reshape(-1, 1)).flatten()
        
    except Exception as e:
        print(f"[错误 - 折叠 {fold_number}] SVC + Ridge 实验执行失败: {e}")
        traceback.print_exc()
        return None, None, None, None, None, -1

    print(f"[SVC + Ridge 实验 - 折叠 {fold_number}] === 运行结束 ===")
    return svc_model, ridge_model, train_times, (y_pred_nlos_val, y_pred_error_val_orig), y_pred_nlos_prob_val, pred_time_val
