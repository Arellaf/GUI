import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

def train_dnn_model(
        file,
        target_column,
        epochs=50,
        layers=None,         # список кортежів (neurons, activation)
        dropout=0.1,
        test_size=0.2,
        validation_split=0.2,
        progress_callback=None,
        stop_callback=None,
        return_scalers=False
):
    # --- 1. Завантаження даних ---
    if file.endswith((".xls", ".xlsx")):
        data = pd.read_excel(file)
    elif file.endswith(".csv"):
        data = pd.read_csv(file)
    else:
        raise ValueError("Only .xls, .xlsx, and .csv formats are supported")

    data = data.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')

    # --- 2. Обробка текстових колонок ---
    non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"Знайдено текстові колонки: {non_numeric}. Вони будуть закодовані числовими значеннями.")
        for col in non_numeric:
            data[col] = data[col].astype(str).str.strip().str.lower()
            unique_vals = {val: idx + 1 for idx, val in enumerate(sorted(data[col].unique()))}
            data[col] = data[col].map(unique_vals)
            print(f"🔹 {col}: {unique_vals}")

    # --- 3. Перевірка пропусків ---
    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values. Please clean them before training.")

    # --- 4. Формування ознак і цілі ---
    feature_columns = [col for col in data.columns if col != target_column]
    X = data[feature_columns].values.astype(np.float32)
    y = data[target_column].values.astype(np.float32).reshape(-1, 1)

    # --- 5. Масштабування ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # --- 6. Поділ на train/test ---
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # --- 7. Архітектура мережі ---
    if layers is None:
        layers = [(64, "relu"), (64, "relu")]  # default deep model

    model = tf.keras.Sequential()
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)
    model.add(normalizer)

    for neurons, activation in layers:
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation="linear"))

    # --- 8. Компіляція ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_absolute_error"
    )

    # --- 9. Callback для прогресу ---
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback.emit(int((epoch + 1) / epochs * 100))
            if stop_callback and stop_callback():
                print("Training stopped by user.")
                self.model.stop_training = True

    callbacks = [ProgressCallback()]

    # --- 10. Навчання ---
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=validation_split,
        verbose=0,
        callbacks=callbacks
    )

    # --- 11. Оцінка ---
    y_pred = model.predict(x_test, verbose=0)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred).flatten()
    y_test_rescaled = scaler_y.inverse_transform(y_test).flatten()

    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))

    print(f"Test MAE: {mae:.4f} | R²: {r2:.4f} | MSE: {mse:.4f}")

    if return_scalers:
        return model, mae, history, x_test, y_test_rescaled, y_pred_rescaled, X, r2, scaler_X, scaler_y
    else:
        return model, mae, history, x_test, y_test_rescaled, y_pred_rescaled, X, r2, scaler_y