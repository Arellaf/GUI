from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from layers.FuzzyLayer import FuzzyLayer
from layers.DefuzzyLayer import DefuzzyLayer


def neurofuzzy_model(filepath, target_column,
                     epochs=50, test_size=0.2,
                     custom_layers=None,
                     fuzzy_output=8,
                     result_size=4,
                     normalize_target_scale=100.0,
                     plot_path=None):

    if filepath.endswith(".xlsx"):
        data = pd.read_excel(filepath, engine="openpyxl")
    elif filepath.endswith(".xls"):
        data = pd.read_excel(filepath, engine="xlrd")
    else:
        data = pd.read_csv(filepath)

    if target_column not in data.columns:
        raise ValueError(f"У файлі немає колонки '{target_column}'")

    X = data.drop(columns=[target_column]).select_dtypes(include=[np.number])
    y = data[target_column].values.astype(np.float32)

    if X.shape[1] == 0:
        raise ValueError("У файлі немає числових ознак.")

    X_norm = X / (X.max(axis=0).replace(0, 1))
    y_norm = y / normalize_target_scale

    x_train, x_test, y_train, y_test = train_test_split(
        X_norm.values, y_norm, test_size=test_size
    )

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1],)))

    model.add(FuzzyLayer(output_dim=fuzzy_output, name="fuzzy"))
    model.add(tf.keras.layers.Flatten())

    if custom_layers:
        for idx, layer_conf in enumerate(custom_layers):
            units = int(layer_conf.get("units", 16))
            activation = layer_conf.get("activation", "relu")
            model.add(tf.keras.layers.Dense(units, activation=activation))

    model.add(DefuzzyLayer(output_dim=1, name="defuzzy"))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.fit(x_train, y_train, epochs=epochs, verbose=0)

    loss, mae = model.evaluate(x_test, y_test, verbose=0)

    y_pred_all = model.predict(x_test).flatten() * normalize_target_scale
    y_real_all = (y_test * normalize_target_scale).flatten()

    result_size = min(result_size, len(y_test))
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
        f"Neuro-Fuzzy навчання виконано за {epochs} епох\n"
        f"test_size={test_size}\n"
        f"MSE: {loss:.4f}\n"
        f"MAE: {mae:.4f}\n"
        f"Точність: {accuracy:.2f}%\n\n"
        f"Прогнози: {np.array2string(predictions, precision=3)}\n"
        f"Реальні: {np.array2string(real_values[:result_size], precision=1)}"
    )

    if plot_path is None:
        plot_path = os.path.join(os.getcwd(), "neurofuzzy_plot.png")

    try:
        plt.figure(figsize=(6,4))
        plt.plot(y_real_all, label="real")
        plt.plot(y_pred_all, label="pred")
        plt.title("Real vs Pred")
        plt.xlabel("index (Y)")
        plt.ylabel("value (X)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    except Exception:
        plot_path = None

    return result_text, model, y_real_all, y_pred_all, plot_path
