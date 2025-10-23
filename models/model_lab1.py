import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def regression_model(filepath="../data/student_scores.xlsx", epochs=50, test_size=0.2, layers=None, result_size=4):
    # === Завантаження даних ===
    data = pd.read_excel(filepath, engine="openpyxl")

    X = data[["hours_studied", "attendance", "assignments_completed"]].values
    y = data["final_score"].values

    # Нормалізація
    X = X / X.max(axis=0)
    y = y / 100.0

    # Поділ на train/test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # === Побудова моделі ===
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(3,)))

    if layers:
        for layer in layers:
            model.add(tf.keras.layers.Dense(layer["units"], activation=layer["activation"]))
    else:
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x_train, y_train, epochs=epochs, verbose=0)

    # === Оцінка (по тесту) ===
    loss, mae = model.evaluate(x_test, y_test, verbose=0)

    # === Один раз рахуємо прогнози ===
    # y_pred_all = model.predict(X).flatten() * 100
    # y_real_all = y * 100
    y_pred_all = model.predict(x_test).flatten() * 100
    y_real_all = y_test * 100

    # result_size = min(result_size, len(X))
    # predictions = y_pred_all[:result_size]
    # real_values = y_real_all[:result_size]
    result_size = min(result_size, len(x_test))
    predictions = y_pred_all[:result_size]
    real_values = y_real_all[:result_size]

    # === Точність (по тесту) ===
    y_pred_test = model.predict(x_test).flatten() * 100
    y_real_test = y_test * 100
    mape = np.mean(np.abs((y_real_test - y_pred_test) / y_real_test)) * 100
    accuracy = 100 - mape

    # === Форматований текст ===
    result_text = (
        f"Навчання виконано на {epochs} епохах\n"
        f"вибірка навчання/тести {test_size}\n"
        f"Втрати (MSE): {loss:.4f}\n"
        f"Середня абсолютна похибка (MAE): {mae:.4f}\n"
        f"Точність моделі (по тесту): {accuracy:.2f}%\n\n"
        f"Прогнози (перші {result_size}): "
        f"{np.array2string(predictions, precision=3, floatmode='fixed', separator=', ')}\n"
        f"Реальні значення: "
        f"{np.array2string(real_values, precision=1, floatmode='fixed', separator=', ')}"
    )

    # вивід в консоль
    print(result_text)

    return result_text, model
