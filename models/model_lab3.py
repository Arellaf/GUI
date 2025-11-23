# File: models/model_lab3.py
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# import custom layers from layers package
from layers.FuzzyLayer import FuzzyLayer
from layers.DefuzzyLayer import DefuzzyLayer


def neurofuzzy_model(filepath, target_column,
                     epochs=50, test_size=0.2, fuzzy_layers=None,
                     result_size=4, normalize_target_scale=100.0,
                     plot_path=None):
    # Load data
    if filepath.endswith((".xlsx", ".xls")):
        data = pd.read_excel(filepath, engine="openpyxl")
    else:
        data = pd.read_csv(filepath)

    if target_column not in data.columns:
        raise ValueError(f"В файле нет колонки '{target_column}'")

    X = data.drop(columns=[target_column]).select_dtypes(include=[np.number])
    y = data[target_column].values.astype(np.float32)

    if X.shape[1] == 0:
        raise ValueError("В файле нет числовых признаков для обучения.")

    # Normalize
    X_norm = X / (X.max(axis=0).replace(0, 1))
    y_norm = y / normalize_target_scale

    x_train, x_test, y_train, y_test = train_test_split(X_norm.values, y_norm, test_size=test_size)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1],)))

    # If fuzzy_layers provided, add them; otherwise default to single fuzzy layer
    if fuzzy_layers:
        for idx, layer_conf in enumerate(fuzzy_layers):
            out_dim = int(layer_conf.get("output_dim", 8))
            model.add(FuzzyLayer(output_dim=out_dim, name=f"fuzzy_{idx}"))
            model.add(tf.keras.layers.Flatten())
    else:
        model.add(FuzzyLayer(output_dim=8, name="fuzzy_default"))
        model.add(tf.keras.layers.Flatten())

    # Defuzzy: aggregate fuzzy outputs into scalar(s)
    # Use output_dim=1 for single regression target
    model.add(DefuzzyLayer(output_dim=1, name="defuzzy"))

    # If defuzzy returns shape (batch,1) — keep it
    # Optionally add a small dense head for further learning
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x_train, y_train, epochs=epochs, verbose=0)

    loss, mae = model.evaluate(x_test, y_test, verbose=0)

    y_pred_all = model.predict(x_test).flatten() * normalize_target_scale
    y_real_all = (y_test * normalize_target_scale).flatten()

    result_size = min(result_size, len(x_test))
    predictions = y_pred_all[:result_size]
    real_values = y_real_all[:result_size]

    mask = y_real_all != 0
    if mask.sum() == 0:
        mape = float('nan')
        accuracy = float('nan')
    else:
        mape = np.mean(np.abs((y_real_all[mask] - y_pred_all[mask]) / y_real_all[mask])) * 100
        accuracy = 100 - mape

    result_text = (
        f"Neuro-Fuzzy обучение выполнено за {epochs} эпох\n"
        f"test_size={test_size}\n"
        f"MSE (loss): {loss:.4f}\n"
        f"MAE: {mae:.4f}\n"
        f"Примерная точность (по MAPE): {accuracy:.2f}%\n\n"
        f"Прогнозы (первые {result_size}): "
        f"{np.array2string(predictions, precision=3, floatmode='fixed', separator=', ')}\n"
        f"Реальные значения: {np.array2string(real_values[:result_size], precision=1, floatmode='fixed', separator=', ')}"
    )

    # plot
    if plot_path is None:
        plot_path = os.path.join(os.getcwd(), "neurofuzzy_plot.png")
    try:
        plt.figure(figsize=(6,4))
        plt.plot(y_real_all, label="real")
        plt.plot(y_pred_all, label="pred")
        plt.title("Real vs Pred (test set)")
        plt.xlabel("index")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    except Exception:
        plot_path = None

    return result_text, model, y_real_all, y_pred_all, plot_path





