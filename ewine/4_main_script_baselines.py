# -*- coding: utf-8 -*-
"""
最終完整校對版主腳本 (v7.3 - 修复CV泄漏/统一指标命名/调整FLOS与Scaler拟合时机)
- 交叉验证中的 scaler 与 FLOS-GMM 改为“每 fold 拟合”，避免数据泄漏
- CV 阶段不再加载/保存 DualTransformer 的内部 scaler
- 最终训练阶段才保存中央 scaler、FLOS-GMM 与 DT 专用 scaler
- 汇总时统一指标命名为 snake_case，便于消融脚本读取
"""
import argparse, random, numpy as np, pandas as pd, tensorflow as tf, traceback
import sys
from pathlib import Path
from scipy.io import savemat
import time, joblib, os, warnings, gc, sys, logging, datetime
from collections import defaultdict
from tensorflow.keras.callbacks import EarlyStopping
# --- 开关 ---
FORCE_CPU_USAGE = False

# --- 导入配置 ---
try:
    import config_v7 as config
    print("[INFO] Loaded configuration from config_v7.py")
except ImportError:
    try:
        import config
        print("[INFO] Loaded configuration from config.py")
    except ImportError:
        print("❌ [错误] 找不到配置文件（config_v7.py / config.py）")
        sys.exit(1)

# --- 自有模块 ---
from data_utils import (load_data, clean_data, map_labels_and_check,
                        prepare_numpy_arrays, split_data, calculate_class_weights)
import flos_module
# --- 放在 main_script.py 顶部原有 import 附近，替换原来的 from model_definition import ... ---
try:
    from model_definition import DualChannelTransformerModel as DualTransformerClass
except Exception:
    try:
        from model_definition import DualTransformerModel as DualTransformerClass
    except Exception:
        from model_definition import DualTransformer as DualTransformerClass
from evaluation_utils import evaluate_classification, evaluate_regression
from svm_experiment import run_svm_ridge_fold_experiment
from dnn_experiment import run_dnn_fold_experiment, build_dnn_model
from single_transformer_experiment import run_single_transformer_fold_experiment, build_single_transformer_model
from xgboost_experiment import run_xgboost_fold_experiment
from cnn_lstm_experiment import run_cnn_lstm_fold_experiment, build_cnn_lstm_model


def setup_logging(save_dir: Path):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    log_file = save_dir / "execution_main.log"
    fh = logging.FileHandler(log_file); fh.setFormatter(log_formatter); root_logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(log_formatter); root_logger.addHandler(ch)
    logging.info(f"日志系统已初始化。日志文件: {log_file}")

def set_global_seed(seed: int):
    print(f"--- [REPRODUCIBILITY] Setting global seed to: {seed} ---")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def _canon_keys(d: dict) -> dict:
    """把指标键名统一成 snake_case 小写，便于后续聚合/消融读取。"""
    out = {}
    for k, v in d.items():
        kk = k.replace(" ", "_").replace("-", "_").lower()
        out[kk] = v
    return out

def main():
    # 1) 解析参数
    parser = argparse.ArgumentParser(description='UWB NLOS/Error Correction Script with More Models')
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--run_only", type=str, choices=["DualTransformer", "SVM", "DNN", "SingleTransformer", "XGBoost", "CNNLSTM"])

    # Transformer 超参覆盖
    parser.add_argument("--transformer_layers", type=int)
    parser.add_argument("--attention_heads", type=int)
    parser.add_argument("--loss_weight_error", type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--ff_dim', type=int)
    
    # 通用深度学习参数
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--optimizer', type=str) # 虽然目前代码里没用上，但为了接口统一先加上
    
    # Early Stopping相关参数
    parser.add_argument('--early_stopping', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--es_monitor', type=str)
    parser.add_argument('--es_patience', type=int)
    parser.add_argument('--es_min_delta', type=float)

    # 其他模型覆盖
    parser.add_argument("--dnn_lr", type=float)
    parser.add_argument("--dnn_dropout", type=float)
    # [旧参数, 已被下面更详细的 --svm_C 替代]
    # parser.add_argument("--svm_c", type=float) 
    parser.add_argument("--ridge_alpha", type=float)

    # DNN 专属参数
    parser.add_argument("--dense_units", type=str, help='Number of units in dense layers, e.g., "128" or "128,64"')
    parser.add_argument("--weight_decay", type=float, help='Weight decay for the optimizer')
    parser.add_argument("--activation", type=str, help='Activation function for dense layers')
    
    # ======================================================================
    # [ADDED] 为 LS-SVM, XGBoost, CNN-LSTM 添加所有缺失的参数定义
    # ======================================================================

    # --- for LS-SVM ---
    parser.add_argument("--svm_kernel", type=str, help='Kernel type for SVM')
    parser.add_argument("--svm_C", type=float, help='Regularization parameter C for SVM')
    parser.add_argument("--svm_gamma", type=str, help='Kernel coefficient for "rbf". Can be "scale" or a float.')
    parser.add_argument("--svm_class_weight", type=str, help='Class weight for SVM')
    parser.add_argument("--standardize", type=lambda x: (str(x).lower() == 'true'), help='Whether to standardize data for SVM')

    # --- for XGBoost ---
    parser.add_argument("--xgb_n_estimators", type=int, help='Number of boosting rounds for XGBoost')
    parser.add_argument("--xgb_max_depth", type=int, help='Maximum tree depth for XGBoost')
    parser.add_argument("--xgb_learning_rate", type=float, help='Learning rate for XGBoost')
    parser.add_argument("--xgb_subsample", type=float, help='Subsample ratio of the training instance')
    parser.add_argument("--xgb_colsample_bytree", type=float, help='Subsample ratio of columns when constructing each tree')
    parser.add_argument("--xgb_gamma", type=float, help='Minimum loss reduction required to make a further partition')
    parser.add_argument("--xgb_min_child_weight", type=int, help='Minimum sum of instance weight needed in a child')

    # --- for CNN-LSTM ---
    parser.add_argument("--conv_filters", type=int, help='Number of filters in the CNN layer')
    parser.add_argument("--kernel_size", type=int, help='Size of the convolution kernel')
    parser.add_argument("--pool_size", type=int, help='Size of the max pooling window')
    parser.add_argument("--lstm_units", type=int, help='Number of units in the LSTM layer')
    
    # ======================================================================
    # [补全结束]
    # ======================================================================

    # 消融强度
    parser.add_argument('--lambda_mask', type=float, default=1.0)
    parser.add_argument('--lambda_ga', type=float, default=1.0)

    args = parser.parse_args()



    # 2) 应用参数/日志
    if args.seed is not None: config.RANDOM_SEED = args.seed
    set_global_seed(config.RANDOM_SEED)

    save_dir_path = Path(args.save_dir) if args.save_dir else config.SAVE_DIRECTORY
    save_dir_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir_path)
    config.SAVE_DIRECTORY = save_dir_path

    if args.run_only:
        logging.info(f"仅运行模型: {args.run_only}")
        config.RUN_DUAL_TRANSFORMER  = (args.run_only == "DualTransformer")
        config.RUN_SVM               = (args.run_only == "SVM")
        config.RUN_DNN               = (args.run_only == "DNN")
        config.RUN_SINGLE_TRANSFORMER= (args.run_only == "SingleTransformer")
        config.RUN_XGBOOST           = (args.run_only == "XGBoost")
        config.RUN_CNN_LSTM          = (args.run_only == "CNNLSTM")

# ======================================================================
# [MODIFIED] 统一应用所有从命令行传入的参数，覆盖config.py的默认值
# ======================================================================

# --- 应用通用深度学习参数 ---
    if args.batch_size is not None:
        config.DUAL_TRANSFORMER_BATCH_SIZE = args.batch_size
        config.DNN_BATCH_SIZE = args.batch_size
        config.SINGLE_TRANSFORMER_BATCH_SIZE = args.batch_size
        config.CNN_LSTM_BATCH_SIZE = args.batch_size
    if args.max_epochs is not None:
        config.DUAL_TRANSFORMER_TRAINING_EPOCHS = args.max_epochs
        config.DNN_TRAINING_EPOCHS = args.max_epochs
        config.SINGLE_TRANSFORMER_TRAINING_EPOCHS = args.max_epochs
        config.CNN_LSTM_TRAINING_EPOCHS = args.max_epochs

    # --- 应用Early Stopping参数 ---
    if args.early_stopping is not None:     config.EARLY_STOPPING_ENABLED = args.early_stopping
    if args.es_monitor is not None:         config.ES_MONITOR = args.es_monitor
    if args.es_patience is not None:        config.ES_PATIENCE = args.es_patience
    if args.es_min_delta is not None:       config.ES_MIN_DELTA = args.es_min_delta

    # --- 应用Transformer & SingleTransformer共享参数 ---
    if args.transformer_layers is not None:
        config.DUAL_TRANSFORMER_MODEL_PARAMS['transformer_layers'] = args.transformer_layers
        config.SINGLE_TRANSFORMER_MODEL_PARAMS['transformer_layers'] = args.transformer_layers
    if args.attention_heads is not None:
        config.DUAL_TRANSFORMER_MODEL_PARAMS['attention_heads'] = args.attention_heads
        config.SINGLE_TRANSFORMER_MODEL_PARAMS['attention_heads'] = args.attention_heads
    if args.ff_dim is not None:
        config.DUAL_TRANSFORMER_MODEL_PARAMS['ff_dim'] = args.ff_dim
        config.SINGLE_TRANSFORMER_MODEL_PARAMS['ff_dim'] = args.ff_dim
    if args.dropout_rate is not None:
        config.DUAL_TRANSFORMER_MODEL_PARAMS['dropout_rate'] = args.dropout_rate
        config.SINGLE_TRANSFORMER_MODEL_PARAMS['dropout_rate'] = args.dropout_rate
        config.DNN_MODEL_PARAMS['dropout_rate'] = args.dropout_rate # DNN也用
        config.CNN_LSTM_MODEL_PARAMS['dropout_rate'] = args.dropout_rate # CNN-LSTM也用
    if args.loss_weight_error is not None:
        config.DUAL_TRANSFORMER_LOSS_WEIGHTS['error_output'] = args.loss_weight_error
        config.SINGLE_TRANSFORMER_LOSS_WEIGHTS['error_output'] = args.loss_weight_error
        config.DNN_LOSS_WEIGHTS['error_output'] = args.loss_weight_error # DNN也用
        config.CNN_LSTM_LOSS_WEIGHTS['error_output'] = args.loss_weight_error # CNN-LSTM也用
    if args.learning_rate is not None:
        config.DUAL_TRANSFORMER_LEARNING_RATE = args.learning_rate
        config.SINGLE_TRANSFORMER_LEARNING_RATE = args.learning_rate
        config.DNN_LEARNING_RATE = args.learning_rate # DNN也用
        config.CNN_LSTM_LEARNING_RATE = args.learning_rate # CNN-LSTM也用

    # --- 应用DNN专属参数 ---
    if args.dense_units is not None:
        config.DNN_MODEL_PARAMS['dense_units'] = args.dense_units
        config.CNN_LSTM_MODEL_PARAMS['dense_units'] = args.dense_units # CNN-LSTM也用
    if args.weight_decay is not None:
        config.DNN_MODEL_PARAMS['weight_decay'] = args.weight_decay
    if args.activation is not None:
        config.DNN_MODEL_PARAMS['activation'] = args.activation
        config.CNN_LSTM_MODEL_PARAMS['activation'] = args.activation # CNN-LSTM也用

    # --- 应用LS-SVM参数 ---
    if args.svm_kernel is not None:
        config.SVC_PARAMS['kernel'] = args.svm_kernel
    if args.svm_C is not None:
        config.SVC_PARAMS['C'] = args.svm_C
    if args.svm_gamma is not None:
        try:
            config.SVC_PARAMS['gamma'] = float(args.svm_gamma)
        except ValueError:
            config.SVC_PARAMS['gamma'] = args.svm_gamma
    if args.svm_class_weight is not None:
        config.SVC_PARAMS['class_weight'] = args.svm_class_weight
    if args.ridge_alpha is not None:
        config.RIDGE_PARAMS['alpha'] = args.ridge_alpha

    # --- 应用XGBoost参数 ---
    if args.xgb_n_estimators is not None:
        config.XGBOOST_CLS_PARAMS['n_estimators'] = args.xgb_n_estimators
        config.XGBOOST_REG_PARAMS['n_estimators'] = args.xgb_n_estimators
    if args.xgb_max_depth is not None:
        config.XGBOOST_CLS_PARAMS['max_depth'] = args.xgb_max_depth
        config.XGBOOST_REG_PARAMS['max_depth'] = args.xgb_max_depth
    if args.xgb_learning_rate is not None:
        config.XGBOOST_CLS_PARAMS['learning_rate'] = args.xgb_learning_rate
        config.XGBOOST_REG_PARAMS['learning_rate'] = args.xgb_learning_rate
    if args.xgb_subsample is not None:
        config.XGBOOST_CLS_PARAMS['subsample'] = args.xgb_subsample
        config.XGBOOST_REG_PARAMS['subsample'] = args.xgb_subsample
    if args.xgb_colsample_bytree is not None:
        config.XGBOOST_CLS_PARAMS['colsample_bytree'] = args.xgb_colsample_bytree
        config.XGBOOST_REG_PARAMS['colsample_bytree'] = args.xgb_colsample_bytree
    if args.xgb_gamma is not None:
        config.XGBOOST_CLS_PARAMS['gamma'] = args.xgb_gamma
        config.XGBOOST_REG_PARAMS['gamma'] = args.xgb_gamma
    if args.xgb_min_child_weight is not None:
        config.XGBOOST_CLS_PARAMS['min_child_weight'] = args.xgb_min_child_weight
        config.XGBOOST_REG_PARAMS['min_child_weight'] = args.xgb_min_child_weight

    # --- 应用CNN-LSTM专属参数 ---
    if args.conv_filters is not None:
        config.CNN_LSTM_MODEL_PARAMS['conv_filters'] = args.conv_filters
    if args.kernel_size is not None:
        config.CNN_LSTM_MODEL_PARAMS['kernel_size'] = args.kernel_size
    if args.pool_size is not None:
        config.CNN_LSTM_MODEL_PARAMS['pool_size'] = args.pool_size
    if args.lstm_units is not None:
        config.CNN_LSTM_MODEL_PARAMS['lstm_units'] = args.lstm_units

# ======================================================================
# [参数应用补全结束]
# ======================================================================

# [ADDED] 创建Early Stopping回调的逻辑
    callbacks = []
    if hasattr(config, 'EARLY_STOPPING_ENABLED') and config.EARLY_STOPPING_ENABLED:
        logging.info(f"EarlyStopping enabled: monitor='{config.ES_MONITOR}', patience={config.ES_PATIENCE}")
        early_stopping_callback = EarlyStopping(
            monitor=config.ES_MONITOR,
            patience=config.ES_PATIENCE,
            min_delta=config.ES_MIN_DELTA,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping_callback)

    # 设备
    if FORCE_CPU_USAGE:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logging.warning("=" * 20 + " CPU USAGE FORCED " + "=" * 20)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and not FORCE_CPU_USAGE:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"[Device] {len(gpus)} GPU(s) found. Using GPU.")
        except RuntimeError as e:
            logging.warning(f"[Device] Mem growth warn: {e}")
    else:
        logging.info("[Device] Using CPU.")

    # 3) 数据
    logging.info("\n--- [Step 1] Loading data ---")
    # [CHANGED] 兼容字符串路径
    rel_paths = [Path(p) for p in config.RELATIVE_DATA_PATHS]
    all_dataframes = [load_data(p) for p in rel_paths if p.exists()]
    if not all_dataframes:
        logging.error("[Fatal] No valid data loaded. Check 'RELATIVE_DATA_PATHS'.")
        sys.exit(1)
    data = pd.concat(all_dataframes, ignore_index=True) if len(all_dataframes) > 1 else all_dataframes[0]
    logging.info(f"Data loaded. Initial shape: {data.shape}")

    logging.info("\n--- [Step 2] Cleaning data ---")
    data = clean_data(data, config.INITIAL_CIR_FEATURES + ['NLOS', 'error'], config.INITIAL_CIR_FEATURES, config.Z_SCORE_THRESHOLD)
    if data is None or data.empty:
        logging.error("[Fatal] Data empty after cleaning."); sys.exit()

    target_nlos_numeric = map_labels_and_check(data['NLOS'])
    if target_nlos_numeric is None:
        logging.error("[Fatal] Label mapping failed."); sys.exit()

    X_cir_full, y_nlos_full, y_error_full_mm = prepare_numpy_arrays(data[config.INITIAL_CIR_FEATURES], target_nlos_numeric, data['error'])
    logging.info(f"Arrays ready. X:{X_cir_full.shape}, y_cls:{y_nlos_full.shape}, y_reg:{y_error_full_mm.shape}")

    # 4) 切分
    logging.info("\n--- [Step 3] Train/Test split ---")
    split_results = split_data(X_cir_full, y_nlos_full, y_error_full_mm, config.TEST_SET_SIZE, config.RANDOM_SEED)
    if split_results is None:
        logging.error("[Fatal] Split failed."); sys.exit()
    X_train_cir_init, X_test_cir, y_train_nlos_init, y_test_nlos, y_train_error_init_mm, y_test_error_mm = split_results

    # 5) K折交叉验证（每fold拟合 scaler + FLOS-GMM）  —— 修复泄漏
    logging.info(f"\n{'='*25} Starting {config.K_FOLDS}-Fold Cross-Validation {'='*25}")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    fold_metrics = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_cir_init, y_train_nlos_init)):
        fold_num = fold + 1
        logging.info(f"\n--- Processing Fold {fold_num}/{config.K_FOLDS} ---")

        X_tr, X_va = X_train_cir_init[train_idx], X_train_cir_init[val_idx]
        y_tr_cls, y_va_cls = y_train_nlos_init[train_idx], y_train_nlos_init[val_idx]
        y_tr_err, y_va_err = y_train_error_init_mm[train_idx], y_train_error_init_mm[val_idx]
        class_weights_fold = calculate_class_weights(y_tr_cls)

        # [CHANGED] 每fold 拟合 scaler
        scaler_cir_fold  = StandardScaler().fit(X_tr)
        scaler_err_fold  = StandardScaler().fit(y_tr_err.reshape(-1,1))
        X_tr_scaled = scaler_cir_fold.transform(X_tr)
        X_va_scaled = scaler_cir_fold.transform(X_va)
        y_tr_err_scaled = scaler_err_fold.transform(y_tr_err.reshape(-1,1)).ravel()
        y_va_err_scaled = scaler_err_fold.transform(y_va_err.reshape(-1,1)).ravel()

        # === Dual Channel Transformer（每fold 拟合 FLOS-GMM，且不load内部scaler）===
        if config.RUN_DUAL_TRANSFORMER:
            model_key = "DualTransformer"
            logging.info(f"     --- Running Fold {fold_num} for {model_key} ---")
            tf.keras.backend.clear_session()
            try:
                # [CHANGED] 每fold 拟合 FLOS-GMM
                ok = flos_module.fit_flos_gmm_models(
                    X_tr, y_tr_cls,
                    max_components_search=config.GMM_MAX_COMPONENTS_SEARCH,
                    covariance_type=config.GMM_COVARIANCE_TYPE,
                    random_state=config.RANDOM_SEED + fold_num
                )
                if not ok: raise RuntimeError("FLOS GMM fit failed (fold).")
                X_tr_flos = flos_module.calculate_flos_features_gmm(X_tr)
                X_va_flos = flos_module.calculate_flos_features_gmm(X_va)
                if X_tr_flos is None or X_va_flos is None:
                    raise RuntimeError("FLOS feature calc failed (fold).")

                dual_model = DualTransformerClass(
                    num_cir_features=X_tr.shape[1], num_flos_features=X_tr_flos.shape[1],
                    model_params=config.DUAL_TRANSFORMER_MODEL_PARAMS,
                    learning_rate=config.DUAL_TRANSFORMER_LEARNING_RATE,
                    loss_weights=config.DUAL_TRANSFORMER_LOSS_WEIGHTS
                )
                dual_model.lambda_mask = args.lambda_mask
                dual_model.lambda_ga   = args.lambda_ga
                dual_model.compile(optimizer="adam")

                # [CHANGED] 不加载DT内部scaler，让其在 train() 内部用训练子集拟合
                val_pack = (X_va, X_va_flos, y_va_cls, y_va_err)
                history = dual_model.train(
                    X_train_cir=X_tr, X_train_flos=X_tr_flos,
                    y_train_nlos=y_tr_cls, y_train_error=y_tr_err,
                    epochs=config.DUAL_TRANSFORMER_TRAINING_EPOCHS,
                    batch_size=config.DUAL_TRANSFORMER_BATCH_SIZE,
                    class_weight=class_weights_fold, validation_data=val_pack
                )
                epochs_ran = len(history.history['loss'])
                fold_metrics[model_key]['stopped_epoch'].append(epochs_ran)

                prob_val, err_scaled_val = dual_model.predict(X_va, X_va_flos)
                y_pred_cls = (prob_val.flatten() > 0.5).astype(int)
                y_pred_err = dual_model.scaler_error.inverse_transform(err_scaled_val.reshape(-1,1)).ravel()

                eval_cls = _canon_keys(evaluate_classification(y_va_cls, y_pred_cls))
                eval_reg = _canon_keys(evaluate_regression(y_va_err,  y_pred_err))
                fold_run_results = {**eval_cls, **eval_reg}
                logging.info(f"          > Fold {fold_num} Val Results ({model_key}): {fold_run_results}")
                for metric, value in fold_run_results.items():
                    fold_metrics[model_key][metric].append(value)

            except Exception as e:
                logging.error(f"Fold {fold_num} for {model_key} failed: {e}")
                logging.error(traceback.format_exc())

        # === DNN ===
        if config.RUN_DNN:
            model_key = "DNN"
            logging.info(f"     --- Running Fold {fold_num} for {model_key} ---")
            tf.keras.backend.clear_session()
            try:
                _, _, _, preds_val, _, _ = run_dnn_fold_experiment(
                    X_train_cir_scaled=X_tr_scaled, y_train_nlos=y_tr_cls,
                    y_train_error_scaled=y_tr_err_scaled,
                    X_val_cir_scaled=X_va_scaled,  y_val_nlos=y_va_cls,
                    y_val_error_scaled=y_va_err_scaled,
                    scaler_error=scaler_err_fold,  # [CHANGED] 传fold scaler
                    dnn_params=config.DNN_MODEL_PARAMS,
                    epochs=config.DNN_TRAINING_EPOCHS, batch_size=config.DNN_BATCH_SIZE,
                    learning_rate=config.DNN_LEARNING_RATE, loss_weights=config.DNN_LOSS_WEIGHTS,
                    fold_number=fold_num, class_weights_dict=class_weights_fold
                )
                if preds_val:
                    y_pred_cls, y_pred_err = preds_val
                    eval_cls = _canon_keys(evaluate_classification(y_va_cls, y_pred_cls))
                    eval_reg = _canon_keys(evaluate_regression(y_va_err,  y_pred_err))
                    fold_run_results = {**eval_cls, **eval_reg}
                    for k,v in fold_run_results.items(): fold_metrics[model_key][k].append(v)
            except Exception as e:
                logging.error(f"Fold {fold_num} for {model_key} failed: {e}")
                logging.error(traceback.format_exc())

        # === Single Transformer ===
        if config.RUN_SINGLE_TRANSFORMER:
            model_key = "SingleTransformer"
            logging.info(f"     --- Running Fold {fold_num} for {model_key} ---")
            tf.keras.backend.clear_session()
            try:
                _, _, _, preds_val, _, _ = run_single_transformer_fold_experiment(
                    X_train_cir_scaled=X_tr_scaled, y_train_nlos=y_tr_cls,
                    y_train_error_scaled=y_tr_err_scaled,
                    X_val_cir_scaled=X_va_scaled,  y_val_nlos=y_va_cls,
                    y_val_error_scaled=y_va_err_scaled,
                    scaler_error=scaler_err_fold, transformer_params=config.SINGLE_TRANSFORMER_MODEL_PARAMS,
                    epochs=config.SINGLE_TRANSFORMER_TRAINING_EPOCHS, batch_size=config.SINGLE_TRANSFORMER_BATCH_SIZE,
                    learning_rate=config.SINGLE_TRANSFORMER_LEARNING_RATE, loss_weights=config.SINGLE_TRANSFORMER_LOSS_WEIGHTS,
                    fold_number=fold_num, class_weights_dict=class_weights_fold
                )
                if preds_val:
                    y_pred_cls, y_pred_err = preds_val
                    eval_cls = _canon_keys(evaluate_classification(y_va_cls, y_pred_cls))
                    eval_reg = _canon_keys(evaluate_regression(y_va_err,  y_pred_err))
                    for k,v in {**eval_cls, **eval_reg}.items(): fold_metrics[model_key][k].append(v)
            except Exception as e:
                logging.error(f"Fold {fold_num} for {model_key} failed: {e}")
                logging.error(traceback.format_exc())

        # === SVM ===
        if config.RUN_SVM:
            model_key = "SVM"
            logging.info(f"     --- Running Fold {fold_num} for {model_key} ---")
            try:
                _, _, _, preds_val, _, _ = run_svm_ridge_fold_experiment(
                    X_train_cir_scaled=X_tr_scaled, y_train_nlos=y_tr_cls,
                    y_train_error_scaled=y_tr_err_scaled,
                    X_val_cir_scaled=X_va_scaled,
                    scaler_error=scaler_err_fold, svm_params=config.SVC_PARAMS,
                    ridge_params=config.RIDGE_PARAMS, fold_number=fold_num,
                    random_seed=config.RANDOM_SEED + fold_num
                )
                if preds_val:
                    y_pred_cls, y_pred_err = preds_val
                    eval_cls = _canon_keys(evaluate_classification(y_va_cls, y_pred_cls))
                    eval_reg = _canon_keys(evaluate_regression(y_va_err,  y_pred_err))
                    for k,v in {**eval_cls, **eval_reg}.items(): fold_metrics[model_key][k].append(v)
            except Exception as e:
                logging.error(f"Fold {fold_num} for {model_key} failed: {e}")
                logging.error(traceback.format_exc())

        # === XGBoost ===
        if config.RUN_XGBOOST:
            model_key = "XGBoost"
            logging.info(f"     --- Running Fold {fold_num} for {model_key} ---")
            try:
                _, _, _, preds_val, _, _ = run_xgboost_fold_experiment(
                    X_train_cir_scaled=X_tr_scaled, y_train_nlos=y_tr_cls,
                    y_train_error_scaled=y_tr_err_scaled,
                    X_val_cir_scaled=X_va_scaled, scaler_error=scaler_err_fold,
                    xgb_cls_params=config.XGBOOST_CLS_PARAMS, xgb_reg_params=config.XGBOOST_REG_PARAMS,
                    fold_number=fold_num, random_seed=config.RANDOM_SEED + fold_num
                )
                if preds_val:
                    y_pred_cls, y_pred_err = preds_val
                    eval_cls = _canon_keys(evaluate_classification(y_va_cls, y_pred_cls))
                    eval_reg = _canon_keys(evaluate_regression(y_va_err,  y_pred_err))
                    for k,v in {**eval_cls, **eval_reg}.items(): fold_metrics[model_key][k].append(v)
            except Exception as e:
                logging.error(f"Fold {fold_num} for {model_key} failed: {e}")
                logging.error(traceback.format_exc())

        # === CNN-LSTM ===
        if config.RUN_CNN_LSTM:
            model_key = "CNNLSTM"
            logging.info(f"     --- Running Fold {fold_num} for {model_key} ---")
            tf.keras.backend.clear_session()
            try:
                _, _, _, preds_val, _, _ = run_cnn_lstm_fold_experiment(
                    X_train_cir_scaled=X_tr_scaled, y_train_nlos=y_tr_cls,
                    y_train_error_scaled=y_tr_err_scaled,
                    X_val_cir_scaled=X_va_scaled, y_val_nlos=y_va_cls,
                    y_val_error_scaled=y_va_err_scaled,
                    scaler_error=scaler_err_fold, cnn_lstm_params=config.CNN_LSTM_MODEL_PARAMS,
                    epochs=config.CNN_LSTM_TRAINING_EPOCHS, batch_size=config.CNN_LSTM_BATCH_SIZE,
                    learning_rate=config.CNN_LSTM_LEARNING_RATE, loss_weights=config.CNN_LSTM_LOSS_WEIGHTS,
                    fold_number=fold_num, class_weights_dict=class_weights_fold
                )
                if preds_val:
                    y_pred_cls, y_pred_err = preds_val
                    eval_cls = _canon_keys(evaluate_classification(y_va_cls, y_pred_cls))
                    eval_reg = _canon_keys(evaluate_regression(y_va_err,  y_pred_err))
                    for k,v in {**eval_cls, **eval_reg}.items(): fold_metrics[model_key][k].append(v)
            except Exception as e:
                logging.error(f"Fold {fold_num} for {model_key} failed: {e}")
                logging.error(traceback.format_exc())

        gc.collect()

    # 6) 聚合CV结果（snake_case）
    logging.info("\n" + "=" * 20 + " Aggregating CV Results " + "=" * 20)
    cv_summary = {}
    for model_key, m2v in fold_metrics.items():
        cv_summary[model_key] = {}
        for metric, values in m2v.items():
            valid_values = [v for v in values if pd.notna(v)]
            if valid_values:
                cv_summary[model_key][f'{metric}_cv_mean'] = float(np.mean(valid_values))
                cv_summary[model_key][f'{metric}_cv_std']  = float(np.std(valid_values))
    cv_summary_df = pd.DataFrame.from_dict(cv_summary, orient='index')
    cv_summary_df.index.name = 'Model'
    logging.info("\n--- CV Summary ---\n" + cv_summary_df.to_string())

    # 7) 最终训练（此处才拟合/保存 “中央” scaler 与 FLOS-GMM 与 DT 专用 scaler）
    from sklearn.preprocessing import StandardScaler
    logging.info("\n" + "=" * 20 + " Final Model Training " + "=" * 20)
    final_models = {}
    class_weights_init_train = calculate_class_weights(y_train_nlos_init)

    # [CHANGED] 在此时拟合 central scaler 并保存
    scaler_cir_central   = StandardScaler().fit(X_train_cir_init)
    scaler_error_central = StandardScaler().fit(y_train_error_init_mm.reshape(-1, 1))
    joblib.dump(scaler_cir_central,   save_dir_path / config.CENTRAL_CIR_SCALER_FILENAME)
    joblib.dump(scaler_error_central, save_dir_path / config.CENTRAL_ERROR_SCALER_FILENAME)

    # [CHANGED] 最终训练前拟合一次“全训练集”FLOS-GMM并保存
    flos_ok_final = True
    if config.RUN_DUAL_TRANSFORMER:
        flos_ok_final = flos_module.fit_flos_gmm_models(
            X_train_cir_init, y_train_nlos_init,
            max_components_search=config.GMM_MAX_COMPONENTS_SEARCH,
            covariance_type=config.GMM_COVARIANCE_TYPE,
            random_state=config.RANDOM_SEED
        )
        if flos_ok_final:
            flos_module.save_flos_gmm_components(
                save_dir_path, config.FLOS_GMM_SCALER_FILENAME,
                config.GMM_LOS_FILENAME, config.GMM_NLOS_FILENAME
            )
        else:
            logging.error("[Final] FLOS GMM fit failed. Disable DualTransformer for final stage.")
            config.RUN_DUAL_TRANSFORMER = False

    X_train_cir_init_scaled_central = scaler_cir_central.transform(X_train_cir_init)
    y_train_error_init_scaled_central = scaler_error_central.transform(y_train_error_init_mm.reshape(-1, 1)).ravel()

    # --- DualTransformer 最终训练 ---
    if config.RUN_DUAL_TRANSFORMER and flos_ok_final:
        logging.info("\n--- Training final DualTransformer model ---")
        X_train_flos_init = flos_module.calculate_flos_features_gmm(X_train_cir_init)
        tf.keras.backend.clear_session()
        final_dt = DualTransformerClass(

            num_cir_features=X_train_cir_init.shape[1], num_flos_features=X_train_flos_init.shape[1],
            model_params=config.DUAL_TRANSFORMER_MODEL_PARAMS, learning_rate=config.DUAL_TRANSFORMER_LEARNING_RATE,
            loss_weights=config.DUAL_TRANSFORMER_LOSS_WEIGHTS
        )
        final_dt.lambda_mask = args.lambda_mask
        final_dt.lambda_ga   = args.lambda_ga
        final_dt.compile(optimizer="adam")
        # 不加载内部 scaler，让其在 train() 内拟合，然后保存出来
        final_dt.train(
            X_train_cir_init, X_train_flos_init, y_train_nlos_init, y_train_error_init_mm,
            epochs=config.DUAL_TRANSFORMER_TRAINING_EPOCHS, batch_size=config.DUAL_TRANSFORMER_BATCH_SIZE,
            class_weight=class_weights_init_train
        )
        final_models["DualTransformer"] = final_dt
        logging.info("   -> Saving final DualTransformer model...")
        final_dt.save_model(save_dir_path / config.DUAL_TRANSFORMER_FINAL_MODEL_FILE)
        # 保存 DT 内部 scaler
        final_dt.save_scalers(
            save_dir_path / config.MAIN_CIR_SCALER_FILENAME,
            save_dir_path / config.MAIN_FLOS_SCALER_FILENAME,
            save_dir_path / config.MAIN_ERROR_SCALER_FILENAME
        )

    # --- DNN ---
    if config.RUN_DNN:
        logging.info("\n--- Training final DNN model ---")
        tf.keras.backend.clear_session()
        final_dnn = build_dnn_model(
            (X_train_cir_init_scaled_central.shape[1],), config.DNN_MODEL_PARAMS,
            config.DNN_LEARNING_RATE, config.DNN_LOSS_WEIGHTS
        )
        sw = {'nlos_output': np.array([class_weights_init_train.get(l, 1.0) for l in y_train_nlos_init])}
        final_dnn.fit(
            x=X_train_cir_init_scaled_central,
            y={'nlos_output': y_train_nlos_init, 'error_output': y_train_error_init_scaled_central},
            epochs=config.DNN_TRAINING_EPOCHS, batch_size=config.DNN_BATCH_SIZE,
            sample_weight=sw, verbose=1, # 改为1可以看到epoch进度
            callbacks=callbacks
        )
        final_models["DNN"] = final_dnn
        final_dnn.save(save_dir_path / config.DNN_FINAL_MODEL_FILE)

    # --- Single Transformer ---
    if config.RUN_SINGLE_TRANSFORMER:
        logging.info("\n--- Training final SingleTransformer model ---")
        tf.keras.backend.clear_session()
        final_st = build_single_transformer_model(
            X_train_cir_init_scaled_central.shape[1], config.SINGLE_TRANSFORMER_MODEL_PARAMS,
            config.SINGLE_TRANSFORMER_LEARNING_RATE, config.SINGLE_TRANSFORMER_LOSS_WEIGHTS
        )
        sw_st = {'nlos_output': np.array([class_weights_init_train.get(l, 1.0) for l in y_train_nlos_init])}
        final_st.fit(
            x=X_train_cir_init_scaled_central,
            y={'nlos_output': y_train_nlos_init, 'error_output': y_train_error_init_scaled_central},
            epochs=config.SINGLE_TRANSFORMER_TRAINING_EPOCHS, batch_size=config.SINGLE_TRANSFORMER_BATCH_SIZE,
            sample_weight=sw_st, verbose=1,
            callbacks=callbacks
        )
        final_models["SingleTransformer"] = final_st
        final_st.save(save_dir_path / config.SINGLE_TRANSFORMER_FINAL_MODEL_FILE)

    # --- SVM ---
    if config.RUN_SVM:
        logging.info("\n--- Training final SVM model ---")
        from sklearn.svm import SVC
        from sklearn.linear_model import Ridge
        final_svc   = SVC(**config.SVC_PARAMS, random_state=config.RANDOM_SEED, class_weight=class_weights_init_train)
        final_ridge = Ridge(**config.RIDGE_PARAMS)  # random_state对默认solver无效，干脆不传
        final_svc.fit(X_train_cir_init_scaled_central, y_train_nlos_init)
        final_ridge.fit(X_train_cir_init_scaled_central, y_train_error_init_scaled_central)
        final_models["SVM"] = (final_svc, final_ridge)
        joblib.dump(final_svc,  save_dir_path / config.SVC_FINAL_MODEL_FILE)
        joblib.dump(final_ridge, save_dir_path / config.RIDGE_FINAL_MODEL_FILE)

    # --- XGBoost ---
    if config.RUN_XGBOOST:
        logging.info("\n--- Training final XGBoost model ---")
        import xgboost as xgb
        final_xgb_cls = xgb.XGBClassifier(**config.XGBOOST_CLS_PARAMS, random_state=config.RANDOM_SEED,
                                          use_label_encoder=False, eval_metric='logloss')
        final_xgb_reg = xgb.XGBRegressor(**config.XGBOOST_REG_PARAMS, random_state=config.RANDOM_SEED)
        final_xgb_cls.fit(X_train_cir_init_scaled_central, y_train_nlos_init)
        final_xgb_reg.fit(X_train_cir_init_scaled_central, y_train_error_init_scaled_central)
        final_models["XGBoost"] = (final_xgb_cls, final_xgb_reg)
        joblib.dump(final_xgb_cls, save_dir_path / config.XGBOOST_CLS_FINAL_MODEL_FILE)
        joblib.dump(final_xgb_reg, save_dir_path / config.XGBOOST_REG_FINAL_MODEL_FILE)

    # --- CNN-LSTM ---
    if config.RUN_CNN_LSTM:
        logging.info("\n--- Training final CNN-LSTM model ---")
        tf.keras.backend.clear_session()
        final_cnn = build_cnn_lstm_model(
            (X_train_cir_init_scaled_central.shape[1],), config.CNN_LSTM_MODEL_PARAMS,
            config.CNN_LSTM_LEARNING_RATE, config.CNN_LSTM_LOSS_WEIGHTS
        )
        sw_cnn = {'nlos_output': np.array([class_weights_init_train.get(l, 1.0) for l in y_train_nlos_init])}
        X_tr3d = np.expand_dims(X_train_cir_init_scaled_central, axis=-1)
        final_cnn.fit(
            x=X_tr3d,
            y={'nlos_output': y_train_nlos_init, 'error_output': y_train_error_init_scaled_central},
            epochs=config.CNN_LSTM_TRAINING_EPOCHS, batch_size=config.CNN_LSTM_BATCH_SIZE,
            sample_weight=sw_cnn, verbose=0
        )
        final_models["CNNLSTM"] = final_cnn
        final_cnn.save(save_dir_path / config.CNN_LSTM_FINAL_MODEL_FILE)

    # 8) 最终测试
    logging.info("\n" + "="*20 + " Summarizing & Testing " + "="*20)
    final_test_eval_metrics = {}
    final_test_raw_predictions = {}

    X_test_cir_scaled_central = scaler_cir_central.transform(X_test_cir)
    X_test_flos = flos_module.calculate_flos_features_gmm(X_test_cir) if ("DualTransformer" in final_models) else None

    for model_key, final_model in final_models.items():
        logging.info(f"--- Evaluating final {model_key} on test set ---")
        y_pred_nlos_test, y_pred_error_test_mm, y_pred_nlos_prob_test = None, None, None

        if model_key == "DualTransformer":
            prob, err_scaled = final_model.predict(X_test_cir, X_test_flos)
            y_pred_nlos_prob_test = prob.flatten()
            y_pred_nlos_test = (y_pred_nlos_prob_test > 0.5).astype(int)
            y_pred_error_test_mm = final_model.scaler_error.inverse_transform(err_scaled.reshape(-1,1)).ravel()

        elif model_key in ["DNN", "SingleTransformer"]:
            preds = final_model.predict(X_test_cir_scaled_central)
            y_pred_nlos_prob_test = preds[0].flatten()
            y_pred_nlos_test = (y_pred_nlos_prob_test > 0.5).astype(int)
            y_pred_error_test_mm = scaler_error_central.inverse_transform(preds[1].reshape(-1,1)).ravel()

        elif model_key == "SVM":
            svc, ridge = final_model
            y_pred_nlos_test = svc.predict(X_test_cir_scaled_central)
            if getattr(svc, "probability", False):
                y_pred_nlos_prob_test = svc.predict_proba(X_test_cir_scaled_central)[:, 1]
            else:
                y_pred_nlos_prob_test = np.full_like(y_pred_nlos_test, np.nan, dtype=float)
            err_scaled = ridge.predict(X_test_cir_scaled_central)
            y_pred_error_test_mm = scaler_error_central.inverse_transform(err_scaled.reshape(-1,1)).ravel()

        elif model_key == "CNNLSTM":
            X_test_3d = np.expand_dims(X_test_cir_scaled_central, axis=-1)
            preds = final_model.predict(X_test_3d, verbose=0)
            y_pred_nlos_prob_test = preds[0].flatten()
            y_pred_nlos_test = (y_pred_nlos_prob_test > 0.5).astype(int)
            y_pred_error_test_mm = scaler_error_central.inverse_transform(preds[1].reshape(-1,1)).ravel()

        elif model_key == "XGBoost":
            xgb_cls, xgb_reg = final_model
            y_pred_nlos_test = xgb_cls.predict(X_test_cir_scaled_central)
            y_pred_nlos_prob_test = xgb_cls.predict_proba(X_test_cir_scaled_central)[:, 1]
            err_scaled = xgb_reg.predict(X_test_cir_scaled_central)
            y_pred_error_test_mm = scaler_error_central.inverse_transform(err_scaled.reshape(-1,1)).ravel()

        if y_pred_nlos_test is not None and y_pred_error_test_mm is not None:
            eval_cls = _canon_keys(evaluate_classification(y_test_nlos, y_pred_nlos_test))
            eval_reg = _canon_keys(evaluate_regression(y_test_error_mm, y_pred_error_test_mm))
            test_results_dict = {**{f"{k}_test":v for k,v in eval_cls.items()},
                                 **{f"{k}_test":v for k,v in eval_reg.items()}}
            final_test_eval_metrics[model_key] = test_results_dict
            logging.info(f"  -> Test Results ({model_key}): {test_results_dict}")

            final_test_raw_predictions[model_key] = {
                'y_test_nlos': y_test_nlos, 'y_pred_nlos': y_pred_nlos_test,
                'y_pred_nlos_prob': y_pred_nlos_prob_test if y_pred_nlos_prob_test is not None else np.full_like(y_pred_nlos_test, np.nan, dtype=float),
                'y_test_error_mm': y_test_error_mm, 'y_pred_error_mm': y_pred_error_test_mm
            }

    test_summary_df = pd.DataFrame.from_dict(final_test_eval_metrics, orient='index')

    final_summary_df = pd.concat([cv_summary_df, test_summary_df], axis=1)
    final_summary_df.index.name = 'Model'
    logging.info("\n--- Final Combined Summary (CV + Test) ---\n" + final_summary_df.to_string())

    summary_csv_path = save_dir_path / config.SUMMARY_CSV_FILENAME
    final_summary_df.to_csv(summary_csv_path, float_format='%.6f')
    logging.info(f"\n✅ Final combined summary saved to: {summary_csv_path}")

    logging.info("\n--- Saving raw predictions to .mat files ---")
    for model_key, preds in final_test_raw_predictions.items():
        if model_key == "DualTransformer": mat_filename_str = config.DUAL_TRANSFORMER_FINAL_RESULTS_MAT
        elif model_key == "DNN":           mat_filename_str = config.DNN_FINAL_RESULTS_MAT
        elif model_key == "SingleTransformer": mat_filename_str = config.SINGLE_TRANSFORMER_FINAL_RESULTS_MAT
        elif model_key == "SVM":           mat_filename_str = config.SVC_RIDGE_FINAL_RESULTS_MAT
        elif model_key == "XGBoost":       mat_filename_str = config.XGBOOST_FINAL_RESULTS_MAT
        elif model_key == "CNNLSTM":       mat_filename_str = config.CNN_LSTM_FINAL_RESULTS_MAT
        else: mat_filename_str = ""
        if mat_filename_str:
            savemat(save_dir_path / mat_filename_str, {k: np.nan_to_num(v, nan=-999.0) for k,v in preds.items()})
            logging.info(f"   -> Saved {model_key} predictions to: {mat_filename_str}")

    logging.info("\n" + "=" * 60)
    logging.info("=== Main Script Execution Finished ===")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
