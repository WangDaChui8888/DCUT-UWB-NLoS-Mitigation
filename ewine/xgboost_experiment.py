# -*- coding: utf-8 -*-
"""
XGBoost模型实验子程序 (v1.0)
- 包含运行XGBoost分类和回归模型的函数。
- 用于K折交叉验证流程中的每一折。
"""
import xgboost as xgb
import numpy as np
from evaluation_utils import evaluate_classification, evaluate_regression

def run_xgboost_fold_experiment(
    X_train_cir_scaled, y_train_nlos, y_train_error_scaled,
    X_val_cir_scaled, scaler_error,
    xgb_cls_params, xgb_reg_params,
    fold_number, random_seed
):
    """
    在单次K-Fold中训练和评估XGBoost分类器和回归器。

    Args:
        X_train_cir_scaled (np.array): 标准化后的训练CIR特征。
        y_train_nlos (np.array): 训练NLOS标签。
        y_train_error_scaled (np.array): 标准化后的训练误差值。
        X_val_cir_scaled (np.array): 标准化后的验证CIR特征。
        scaler_error (StandardScaler): 用于反标准化误差预测的scaler。
        xgb_cls_params (dict): XGBoost分类器参数。
        xgb_reg_params (dict): XGBoost回归器参数。
        fold_number (int): 当前的折数（用于日志）。
        random_seed (int): 随机种子。

    Returns:
        tuple: 包含训练好的模型和验证集预测结果的元组。
    """
    print(f"      - XGBoost Fold {fold_number}: Training Classifier...")

    # 1. 训练XGBoost分类器
    xgb_classifier = xgb.XGBClassifier(**xgb_cls_params, random_state=random_seed, use_label_encoder=False, eval_metric='logloss')
    xgb_classifier.fit(X_train_cir_scaled, y_train_nlos)

    print(f"      - XGBoost Fold {fold_number}: Training Regressor...")
    # 2. 训练XGBoost回归器
    xgb_regressor = xgb.XGBRegressor(**xgb_reg_params, random_state=random_seed)
    xgb_regressor.fit(X_train_cir_scaled, y_train_error_scaled)

    # 3. 在验证集上进行预测
    print(f"      - XGBoost Fold {fold_number}: Predicting on validation set...")
    y_pred_nlos_val = xgb_classifier.predict(X_val_cir_scaled)
    y_pred_error_scaled_val = xgb_regressor.predict(X_val_cir_scaled)

    # 4. 反标准化误差预测
    y_pred_error_val_mm = scaler_error.inverse_transform(y_pred_error_scaled_val.reshape(-1, 1)).flatten()

    models = (xgb_classifier, xgb_regressor)
    predictions = (y_pred_nlos_val, y_pred_error_val_mm)
    
    # 为了与其他实验函数保持一致的返回结构
    history_cls = None
    history_reg = None
    
    return models, history_cls, history_reg, predictions, None, None
