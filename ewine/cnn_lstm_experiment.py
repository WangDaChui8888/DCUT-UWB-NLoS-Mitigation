# -*- coding: utf-8 -*-
"""
CNN-LSTM模型定义与实验子程序 (v1.0)
- 包含构建、训练和评估CNN-LSTM模型的函数。
- 适用于K折交叉验证流程。
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

def build_cnn_lstm_model(input_shape, model_params, learning_rate, loss_weights):
    """
    构建一个并行的CNN-LSTM模型，用于NLOS分类和误差回归。

    Args:
        input_shape (tuple): 输入特征的形状 (num_features,)。
        model_params (dict): 包含模型架构超参数的字典。
        learning_rate (float): 优化器的学习率。
        loss_weights (dict): 用于两个输出的损失权重。

    Returns:
        tf.keras.Model: 编译好的CNN-LSTM模型。
    """
    # 从参数字典中获取超参数
    conv_filters = model_params.get('conv_filters', 64)
    kernel_size = model_params.get('kernel_size', 3)
    pool_size = model_params.get('pool_size', 2)
    lstm_units = model_params.get('lstm_units', 50)
    dense_units = model_params.get('dense_units', [64])
    dropout_rate = model_params.get('dropout_rate', 0.2)
    activation = model_params.get('activation', 'relu')

    # 输入层
    input_layer = Input(shape=input_shape, name='cir_input')
    
    # Reshape to be 3D for Conv1D [batch, steps, channels]
    # 我们将特征数视为 "steps"，并添加一个 "channels" 维度
    reshaped_input = Reshape((input_shape[0], 1))(input_layer)

    # CNN部分
    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, activation=activation, padding='same')(reshaped_input)
    x = MaxPooling1D(pool_size=pool_size)(x)
    
    # LSTM部分
    x = LSTM(units=lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)(x)

    # 全连接层
    for units in dense_units:
        x = Dense(units, activation=activation)(x)
        x = Dropout(dropout_rate)(x)

    # 输出层
    nlos_output = Dense(1, activation='sigmoid', name='nlos_output')(x)
    error_output = Dense(1, activation='linear', name='error_output')(x)

    # 构建模型
    model = Model(inputs=input_layer, outputs=[nlos_output, error_output])

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'nlos_output': 'binary_crossentropy', 'error_output': 'mean_squared_error'},
        loss_weights=loss_weights,
        metrics={'nlos_output': ['accuracy']}
    )
    return model

def run_cnn_lstm_fold_experiment(
    X_train_cir_scaled, y_train_nlos, y_train_error_scaled,
    X_val_cir_scaled, y_val_nlos, y_val_error_scaled,
    scaler_error, cnn_lstm_params,
    epochs, batch_size, learning_rate, loss_weights,
    fold_number, class_weights_dict
):
    """
    在单次K-Fold中训练和评估CNN-LSTM模型。
    """
    print(f"      - CNN-LSTM Fold {fold_number}: Building model...")
    input_shape = (X_train_cir_scaled.shape[1],)
    
    model = build_cnn_lstm_model(input_shape, cnn_lstm_params, learning_rate, loss_weights)
    
    # Keras 需要 sample_weight 作为字典
    keras_sample_weight = {'nlos_output': np.array([class_weights_dict.get(l, 1.0) for l in y_train_nlos])}
    
    # 为CNN-LSTM准备3D输入数据
    X_train_3d = np.expand_dims(X_train_cir_scaled, axis=-1)
    X_val_3d = np.expand_dims(X_val_cir_scaled, axis=-1)

    print(f"      - CNN-LSTM Fold {fold_number}: Training...")
    history = model.fit(
        x=X_train_3d,
        y={'nlos_output': y_train_nlos, 'error_output': y_train_error_scaled},
        validation_data=(X_val_3d, {'nlos_output': y_val_nlos, 'error_output': y_val_error_scaled}),
        epochs=epochs,
        batch_size=batch_size,
        sample_weight=keras_sample_weight,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
    )

    print(f"      - CNN-LSTM Fold {fold_number}: Predicting on validation set...")
    # 预测
    y_pred_nlos_prob_val, y_pred_error_scaled_val = model.predict(X_val_3d)
    
    # 处理预测结果
    y_pred_nlos_val = (y_pred_nlos_prob_val.flatten() > 0.5).astype(int)
    y_pred_error_val_mm = scaler_error.inverse_transform(y_pred_error_scaled_val.reshape(-1, 1)).flatten()

    predictions = (y_pred_nlos_val, y_pred_error_val_mm)
    
    # 同样，为了保持返回结构一致
    history_cls = history
    history_reg = None

    return model, history_cls, history_reg, predictions, None, None
