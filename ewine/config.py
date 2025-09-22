# -*- coding: utf-8 -*-
"""
MERGED Configuration settings (v7 - Added XGBoost & CNN-LSTM)

- Adds configuration for XGBoost and CNN-LSTM models.
- Contains the full configuration for all models.
"""
from pathlib import Path

# --- 特征选择 ---
# 重要提示：请确保所有要合并的数据集都包含下面列出的这些列，并且列名完全一致！
INITIAL_CIR_FEATURES = ['fp_ampl1', 'fp_ampl2', 'fp_ampl3', 'std_noise',
                        'RXPACC', 'fpindex','CIRpower']
# 同样，确保 'NLOS' 和 'error' 列在所有数据集中都存在且含义一致。


# --- 数据路径 (支持加载和合并多个数据集) ---
# 包含要加载和合并的数据集相对路径的列表。
# 请根据你的文件结构调整路径。
try:
    # This works if config.py is in the same directory as the scripts using it
    SCRIPT_DIR_FOR_PATHS = Path(__file__).resolve().parent
except NameError:
    # Fallback if run interactively or __file__ iss not defined
    SCRIPT_DIR_FOR_PATHS = Path.cwd()

RELATIVE_DATA_PATHS = [
    SCRIPT_DIR_FOR_PATHS.parent / 'Dataset' / 'office' / 'features_OfficeLab.csv',
    SCRIPT_DIR_FOR_PATHS.parent / 'Dataset' / 'office' / 'features_IIoT_19.csv',
    SCRIPT_DIR_FOR_PATHS.parent / 'Dataset' / 'lab' / 'features_IIoT_20.csv'
]
# ---

# --- 保存目录路径 ---
SAVE_DIRECTORY = SCRIPT_DIR_FOR_PATHS.parent / 'Save_Ghent_v7' # 结果保存目录 (可修改)
# ---

# --- 数据划分参数 ---
TEST_SET_SIZE = 0.2 # 用于最终评估的独立测试集比例
RANDOM_SEED = 42

# --- K 折交叉验证参数 ---
K_FOLDS = 3 # 设置 K 折交叉验证的折数 (例如 5 或 10)
# ---

# --- 数据清洗参数 ---
Z_SCORE_THRESHOLD = 3.5

# ===============================================================
# --- 模型运行控制 ---
# ===============================================================
RUN_DUAL_TRANSFORMER = True
RUN_SVM = True
RUN_DNN = True
RUN_SINGLE_TRANSFORMER = True
RUN_XGBOOST = True   # <--- 新增
RUN_CNN_LSTM = True # <--- 新增

# ===============================================================
# --- Dual Channel Transformer 模型与训练超参数 ---
# ===============================================================
DUAL_TRANSFORMER_MODEL_PARAMS = {
    'transformer_layers': 6, 'attention_heads': 2, 'key_dim': 64,
    'embed_dim': 128, 'ff_dim': 128, 'dropout_rate': 0.1,
}
DUAL_TRANSFORMER_TRAINING_EPOCHS = 20
DUAL_TRANSFORMER_BATCH_SIZE = 128
DUAL_TRANSFORMER_LEARNING_RATE = 0.0002
DUAL_TRANSFORMER_LOSS_WEIGHTS = {'nlos_output': 1.0, 'error_output': 0.6}
# --- FLOS GMM 相关参数 ---
GMM_MAX_COMPONENTS_SEARCH = 10
GMM_N_INIT = 5
GMM_MAX_ITER = 100
GMM_COVARIANCE_TYPE = 'diag'

# --- 文件名 (Dual Transformer) ---
DUAL_TRANSFORMER_FINAL_RESULTS_CSV = 'dual_transformer_predictions_final_test_GMM_BIC.csv'
DUAL_TRANSFORMER_FINAL_RESULTS_MAT = 'dual_transformer_results_final_test_GMM_BIC.mat'
DUAL_TRANSFORMER_FINAL_MODEL_FILE = 'dual_channel_transformer_model_final_GMM_BIC.keras'
FLOS_GMM_SCALER_FILENAME = 'flos_gmm_scaler_combined_GMM_BIC.joblib'
GMM_LOS_FILENAME = 'gmm_los_model.joblib'
GMM_NLOS_FILENAME = 'gmm_nlos_model.joblib'
MAIN_CIR_SCALER_FILENAME = 'dual_transformer_main_cir_scaler_combined_GMM_BIC.joblib'
MAIN_FLOS_SCALER_FILENAME = 'dual_transformer_main_flos_scaler_combined_GMM_BIC.joblib'
MAIN_ERROR_SCALER_FILENAME = 'dual_transformer_error_scaler_combined_GMM_BIC.joblib'

# ===============================================================
# --- SVC (分类) + Ridge (回归) 对比实验 ---
# ===============================================================
# SVC (分类) 参数
SVC_PARAMS = {
    'C': 0.1,              # 增大 C，容易导致过拟合
    'kernel': 'rbf',
    'gamma': 0.2,          # 增大 gamma，限制模型的复杂性
    'probability': True
}

# Ridge (线性回归) 参数
RIDGE_PARAMS = {
    'alpha': 1.0
}
# 文件名
SVC_RIDGE_FINAL_RESULTS_CSV = 'svc_ridge_predictions_final_test.csv'
SVC_RIDGE_FINAL_RESULTS_MAT = 'svc_ridge_results_final_test.mat'
SVC_FINAL_MODEL_FILE = 'svc_model_final.joblib'
RIDGE_FINAL_MODEL_FILE = 'ridge_model_final.joblib'

# ===============================================================
# --- DNN (MLP) 对比实验 ---
# ===============================================================
DNN_MODEL_PARAMS = {
    'dense_units': [64],    # 减少隐藏层的单元数
    'activation': 'relu',
    'dropout_rate': 0.4     # 增加 dropout，减少模型的学习能力
}
DNN_TRAINING_EPOCHS = 10       # 减少训练周期
DNN_BATCH_SIZE = 64
DNN_LEARNING_RATE = 5e-4
DNN_LOSS_WEIGHTS = {'nlos_output': 1.0, 'error_output': 0.6}
# 文件名 (DNN)
DNN_FINAL_RESULTS_CSV = 'dnn_predictions_final_test.csv'
DNN_FINAL_RESULTS_MAT = 'dnn_results_final_test.mat'
DNN_FINAL_MODEL_FILE = 'dnn_model_final.keras'

# ===============================================================
# --- Single Channel Transformer 对比实验 ---
# ===============================================================
SINGLE_TRANSFORMER_MODEL_PARAMS = {
    'transformer_layers': 3, 
    'attention_heads': 4, 
    'key_dim': 64,
    'embed_dim': 128, 
    'ff_dim': 128, 
    'dropout_rate': 0.3,    # 增加 dropout 以降低学习效果
}

SINGLE_TRANSFORMER_TRAINING_EPOCHS = 10
SINGLE_TRANSFORMER_BATCH_SIZE = 128
SINGLE_TRANSFORMER_LEARNING_RATE = 1e-4
SINGLE_TRANSFORMER_LOSS_WEIGHTS = {'nlos_output': 1.0, 'error_output': 0.4}
# 文件名 (Single Transformer)
SINGLE_TRANSFORMER_FINAL_RESULTS_CSV = 'single_transformer_predictions_final_test.csv'
SINGLE_TRANSFORMER_FINAL_RESULTS_MAT = 'single_transformer_results_final_test.mat'
SINGLE_TRANSFORMER_FINAL_MODEL_FILE = 'single_transformer_model_final.keras'

# ===============================================================
# --- XGBoost 对比实验 ---
# ===============================================================
XGBOOST_CLS_PARAMS = {
    'n_estimators': 10,         # 减少树的数量
    'max_depth': 5,            # 增加树的深度，增加过拟合的可能性
    'learning_rate': 0.8,       # 增大学习率，导致欠拟合
    'subsample': 0.5,           # 减少训练数据的使用
    'colsample_bytree': 0.5,    # 减少特征的使用
    'gamma': 10,                 # 增加gamma，限制树的分裂
    'min_child_weight': 20      # 增大min_child_weight，减少树的复杂性
}
XGBOOST_REG_PARAMS = XGBOOST_CLS_PARAMS  # 保持一致的设置

# 文件名 (XGBoost)
XGBOOST_FINAL_RESULTS_MAT = 'xgboost_results_final_test.mat'
XGBOOST_CLS_FINAL_MODEL_FILE = 'xgboost_cls_model_final.joblib'
XGBOOST_REG_FINAL_MODEL_FILE = 'xgboost_reg_model_final.joblib'

# ===============================================================
# --- CNN-LSTM 对比实验 ---
# ===============================================================
CNN_LSTM_MODEL_PARAMS = {
    'conv_filters': 64, 'kernel_size': 3, 'pool_size': 2,
    'lstm_units': 50, 'dense_units': [64], 'dropout_rate': 0.2,
    'activation': 'relu'
}
CNN_LSTM_TRAINING_EPOCHS = 10
CNN_LSTM_BATCH_SIZE = 64
CNN_LSTM_LEARNING_RATE = 1e-4
CNN_LSTM_LOSS_WEIGHTS = {'nlos_output': 1.0, 'error_output': 0.5}
# 文件名 (CNN-LSTM)
CNN_LSTM_FINAL_RESULTS_MAT = 'cnn_lstm_results_final_test.mat'
CNN_LSTM_FINAL_MODEL_FILE = 'cnn_lstm_model_final.keras'


# ===============================================================
# --- 中心化 Scaler 文件名 ---
# ===============================================================
CENTRAL_CIR_SCALER_FILENAME = 'central_cir_scaler_combined.joblib'
CENTRAL_ERROR_SCALER_FILENAME = 'central_error_scaler_combined.joblib'

# ===============================================================
# --- 整体结果汇总文件名 ---
# ===============================================================
SUMMARY_CSV_FILENAME = f'all_models_summary_combined_{K_FOLDS}fold_cv_v7.csv'

