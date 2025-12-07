import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

def classification_model(filepath, target_column, epochs=50, test_size=0.2, layers=None, result_size=4):
    # === Завантаження даних ===
    if filepath.endswith(".xlsx"):
        data = pd.read_excel(filepath, engine="openpyxl")
    elif filepath.endswith(".xls"):
        data = pd.read_excel(filepath, engine="xlrd")
    else:
        data = pd.read_csv(filepath)

    if target_column not in data.columns:
        raise ValueError(f"У файлі немає колонки '{target_column}'")

    # === Відокремлення цільової колонки ===
    y = data[target_column]
    X = data.drop(columns=[target_column])

    # === Автоматичне кодування нечислових колонок ===
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
            encoder = LabelEncoder()
            try:
                X[col] = encoder.fit_transform(X[col].astype(str))
            except Exception:
                X[col] = pd.factorize(X[col].astype(str))[0]

    # === Перевірка, що залишились колонки ===
    if X.shape[1] == 0:
        raise ValueError("У файлі немає доступних ознак для навчання (усі порожні).")

    # === Кодування міток ===
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    num_classes = len(target_encoder.classes_)

    # === Масштабування числових ознак ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=test_size, random_state=42)

    # === Побудова моделі ===
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1],)))

    if layers:
        for layer in layers:
            model.add(tf.keras.layers.Dense(layer["units"], activation=layer["activation"]))
    else:
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, verbose=0)

    # === Оцінка ===
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    # === Прогнози ===
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # === Відновлення оригінальних назв класів ===
    result_size = min(result_size, len(y_test))
    preds = target_encoder.inverse_transform(y_pred[:result_size])
    reals = target_encoder.inverse_transform(y_test[:result_size])

    # === Звіт ===
    # report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, zero_division=0)

    # === Форматований текст ===
    result_text = (
        f"Навчання виконано на {epochs} епохах\n"
        f"Вибірка навчання/тести: {1 - test_size:.1f}/{test_size:.1f}\n"
        f"Втрати (loss): {loss:.4f}\n"
        f"Точність моделі (accuracy): {accuracy * 100:.2f}%\n\n"
        f"Класи: {list(target_encoder.classes_)}\n\n"
        f"Прогнози (перші {result_size}): {preds}\n"
        f"Реальні значення: {reals}\n\n"
        # f"=== Детальний звіт ===\n{report}"
    )

    print(result_text)
    return result_text, model
