# -*- coding: utf-8 -*-
"""
DNN (MLP) 对比实验模块 (由 CNN 修改而来)
适配 K 折交叉验证：在指定的训练/验证折叠上训练和预测。
"""
import numpy as np
import pandas as pd
# import joblib # 不再需要在此处保存 scaler
import time
import traceback
# from pathlib import Path # 不再需要在此处处理路径
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.losses import Huber
# from sklearn.preprocessing import StandardScaler # Scaler 由调用者传入

# 假设 evaluation_utils 在可导入路径下 (虽然不再直接调用评估函数)
try:
    from evaluation_utils import evaluate_classification, evaluate_regression
except ImportError:
    print("[警告] dnn_experiment.py: 无法导入 evaluation_utils。")
    # 定义占位符以防万一，但它们不应被调用
    def evaluate_classification(*args, **kwargs): return {}
    def evaluate_regression(*args, **kwargs): return {}

def build_dnn_model(input_shape, dnn_params, learning_rate, loss_weights):
    """构建 DNN (MLP) 模型 (保持不变)"""
    # 从 dnn_params 获取参数
    hidden_units_list = dnn_params.get('dense_units', [128, 64]) # 默认隐藏层结构
    activation = dnn_params.get('activation', 'relu')
    dropout_rate = dnn_params.get('dropout_rate', 0.15) # DNN 可能需要不同的 dropout

    inputs = layers.Input(shape=input_shape, name='dnn_input') # shape (num_features,)

    # 直接连接输入到第一个隐藏层
    x = inputs

    # 添加 DNN 隐藏层
    print(f"[DNN 模型] 添加隐藏层，单元数: {hidden_units_list}, 激活函数: {activation}")
    for units in hidden_units_list:
        x = layers.Dense(units, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x) # 在每个隐藏层后添加 Dropout

    # 输出层 (保持不变)
    nlos_output = layers.Dense(1, activation='sigmoid', name='nlos_output')(x)
    error_output = layers.Dense(1, activation='linear', name='error_output')(x)

    model = models.Model(inputs=inputs, outputs=[nlos_output, error_output], name="DNN_Model") # 更新模型名称

    # 编译 (保持不变)
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss={'nlos_output': 'binary_crossentropy', 'error_output': Huber()},
                  loss_weights=loss_weights,
                  metrics={'nlos_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]}) # 添加了P/R
    print("[DNN 模型] DNN 模型构建和编译完成。")
    model.summary()
    return model

# --- 修改后的函数签名和逻辑以适应 K 折交叉验证 ---
def run_dnn_fold_experiment(
    # --- 训练数据 (当前折叠) ---
    X_train_cir_scaled,   # 已缩放的训练 CIR 特征
    y_train_nlos,         # 训练 NLOS 标签
    y_train_error_scaled, # 已缩放的训练 Error 目标 (1D or 2D, fit 需要 2D)
    # --- 验证数据 (当前折叠) ---
    X_val_cir_scaled,     # 已缩放的验证 CIR 特征
    y_val_nlos,           # 验证 NLOS 标签
    y_val_error_scaled,   # 已缩放的验证 Error 目标 (1D or 2D)
    # --- 逆缩放器 (用于最终预测逆缩放) ---
    scaler_error,         # 拟合好的 Error 缩放器
    # --- 模型和训练参数 ---
    dnn_params: dict,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_weights: dict,
    # --- 其他 ---
    fold_number: int,      # 当前折叠编号 (用于日志)
    class_weights_dict=None # 类别权重 (针对当前训练折叠计算)
    ):
    """
    在 K 折交叉验证的单个折叠上运行 DNN (MLP) 实验。
    训练模型并在验证集上进行预测。

    Args:
        X_train_cir_scaled: 当前训练折叠的缩放后 CIR 特征。
        y_train_nlos: 当前训练折叠的 NLOS 标签。
        y_train_error_scaled: 当前训练折叠的缩放后 Error 目标。
        X_val_cir_scaled: 当前验证折叠的缩放后 CIR 特征。
        y_val_nlos: 当前验证折叠的 NLOS 标签。
        y_val_error_scaled: 当前验证折叠的缩放后 Error 目标。
        scaler_error: 拟合好的 Error StandardScaler (用于逆缩放验证集预测)。
        dnn_params: DNN 模型参数。
        epochs: 训练轮数。
        batch_size: 批处理大小。
        learning_rate: 学习率。
        loss_weights: 损失权重。
        fold_number: 当前折叠的编号 (从 1 开始)。
        class_weights_dict: 当前训练折叠的类别权重。

    Returns:
        tuple: (trained_model, history, train_time, val_predictions, val_probabilities, pred_time_val)
               如果失败则返回 (None, None, -1, None, None, -1)
        其中:
            trained_model: 训练好的 Keras 模型。
            history: Keras 训练历史记录。
            train_time: 训练耗时 (秒)。
            val_predictions: 在验证集上的预测元组 (y_pred_nlos_val, y_pred_error_val_orig)。
            val_probabilities: 在验证集上的 NLOS 预测概率。
            pred_time_val: 在验证集上的预测耗时 (秒)。
    """
    print(f"\n[DNN 实验 - 折叠 {fold_number}] === 开始运行 ===")
    model = None
    history = None
    train_time = -1
    pred_time_val = -1
    y_pred_nlos_val = None
    y_pred_error_val_orig = None
    y_pred_nlos_prob_val = None

    try:
        # --- 数据形状检查 (可选但推荐) ---
        print(f"[DNN 折叠 {fold_number}] 训练数据形状: CIR={X_train_cir_scaled.shape}, NLOS={y_train_nlos.shape}, Error={y_train_error_scaled.shape}")
        print(f"[DNN 折叠 {fold_number}] 验证数据形状: CIR={X_val_cir_scaled.shape}, NLOS={y_val_nlos.shape}, Error={y_val_error_scaled.shape}")

        # --- 模型构建 ---
        input_shape = (X_train_cir_scaled.shape[1],) # (num_features,)
        model = build_dnn_model(input_shape, dnn_params, learning_rate, loss_weights)

        # --- 回调函数 ---
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
        ]

        # --- 样本权重 ---
        keras_sample_weight = None
        if class_weights_dict:
            try:
                sample_weights_nlos = np.array([class_weights_dict.get(label, 1.0) for label in y_train_nlos])
                keras_sample_weight = {'nlos_output': sample_weights_nlos}
                print(f"[DNN 折叠 {fold_number}] 将应用样本权重 (仅用于 NLOS 输出)。")
            except Exception as e: print(f"[警告 - 折叠 {fold_number}] DNN 计算样本权重时出错: {e}")

        # --- 模型训练 ---
        print(f"[DNN 折叠 {fold_number}] 开始训练，共 {epochs} 轮...")
        train_start_time = time.time()
        # 准备验证数据格式
        # Keras 的 validation_data 需要一个元组 (x_val, y_val)
        # y_val 可以是列表或字典，取决于模型输出
        validation_data_prepared = (
            X_val_cir_scaled,
            {'nlos_output': y_val_nlos, 'error_output': y_val_error_scaled} # 确保 Error 目标也是缩放后的
        )

        history_obj = model.fit(
            x=X_train_cir_scaled, # 训练输入
            y={'nlos_output': y_train_nlos, 'error_output': y_train_error_scaled}, # 训练目标 (Error 需缩放)
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data_prepared, # 使用传入的验证集
            callbacks=callbacks_list,
            sample_weight=keras_sample_weight,
            verbose=2
        )
        train_time = time.time() - train_start_time
        history = history_obj.history # 保存训练历史
        print(f"[DNN 折叠 {fold_number}] 训练完成，耗时: {train_time:.2f} 秒")

        # --- 在验证集上预测 ---
        print(f"[DNN 折叠 {fold_number}] 在验证集上进行预测...")
        pred_start_time = time.time()
        predictions_val = model.predict(X_val_cir_scaled)
        pred_time_val = time.time() - pred_start_time
        print(f"[DNN 折叠 {fold_number}] 验证集预测完成，耗时: {pred_time_val:.4f} 秒")

        y_pred_nlos_prob_val = predictions_val[0].flatten() # NLOS 概率
        y_pred_error_scaled_val = predictions_val[1]       # 缩放后的 Error 预测

        # --- 后处理验证集预测 ---
        y_pred_nlos_val = (y_pred_nlos_prob_val > 0.5).astype(int) # 使用 0.5 阈值 (可在外部调整)
        print(f"[DNN 折叠 {fold_number}] 逆缩放验证集预测误差...")
        y_pred_error_val_orig = scaler_error.inverse_transform(y_pred_error_scaled_val.reshape(-1, 1)).flatten()

        # --- 不在此处进行评估或保存 ---
        # 评估将在主循环中完成
        # 模型和最终结果将在主循环结束后保存

    except Exception as e:
        print(f"[错误 - 折叠 {fold_number}] DNN 实验执行失败: {e}")
        traceback.print_exc()
        # 清理并返回失败标志
        tf.keras.backend.clear_session()
        return None, None, -1, None, None, -1 # 返回 None 表示失败

    print(f"[DNN 实验 - 折叠 {fold_number}] === 运行结束 ===")
    # 不清除会话，因为模型需要返回
    # tf.keras.backend.clear_session() # 会话将在主循环中管理

    # 返回训练好的模型和在验证集上的结果
    return model, history, train_time, (y_pred_nlos_val, y_pred_error_val_orig), y_pred_nlos_prob_val, pred_time_val
