import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QPushButton, QSpinBox, QComboBox
)
from tensorflow.keras.models import load_model
from workers.classification_worker import ClassificationWorker
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Model2Page(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.data_path = None
        self.trained_model = None
        self.custom_layers = []
        self.worker = None

        # --- Прив’язуємо кнопки ---
        try:
            self.ui.upload_my_data_1.clicked.connect(self.select_excel_file)
        except Exception:
            pass

        try:
            self.ui.learn_btn_2.clicked.connect(self.run_classification_model)
        except Exception:
            pass

        try:
            self.ui.safe_model_btn_1.clicked.connect(self.save_trained_model)
        except Exception:
            pass

        try:
            self.ui.upload_my_model_1.clicked.connect(self.load_trained_model)
        except Exception:
            pass

        try:
            self.ui.add_layer_btn_1.clicked.connect(self.add_layer)
        except Exception:
            pass

        # --- Перший базовий шар ---
        try:
            self.add_layer()
        except Exception as e:
            print("Не вдалося створити базовий шар:", e)

        # --- Ховаємо target_column спочатку ---
        try:
            self.ui.target_column_1.hide()
        except Exception:
            pass

        # --- Перемикання між сторінками ---
        try:
            self.ui.stackedWidget_btn_next_3.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
            self.ui.stackedWidget_btn_back_4.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
            self.ui.stackedWidget_btn_next_5.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(4))
        except Exception:
            pass

    # ========================== ВИБІР ФАЙЛУ ==========================
    def select_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть ваш Excel-файл", "", "Excel файли (*.xlsx *.xls *.csv)"
        )
        if not file_path:
            return

        self.data_path = file_path
        QMessageBox.information(self, "Файл вибрано", f"Використовується файл:\n{file_path}")

        try:
            if file_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path, nrows=0)
            else:
                df = pd.read_csv(file_path, nrows=0)

            columns = df.columns.tolist()

            if hasattr(self.ui, "target_column_1"):
                self.ui.target_column_1.clear()
                self.ui.target_column_1.addItems(columns)
                self.ui.target_column_1.setEnabled(True)
                self.ui.target_column_1.show()
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося прочитати файл:\n{e}")

    # ========================== ДОДАВАННЯ ШАРІВ ==========================
    def add_layer(self):
        if not hasattr(self.ui, "add_layers_model_2"):
            QMessageBox.warning(self, "UI error", "Віджет add_layers_model_2 не знайдено у UI.")
            return

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 2, 2, 2)

        spin = QSpinBox()
        spin.setRange(1, 4096)
        spin.setValue(128)

        combo = QComboBox()
        combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
        combo.setCurrentText("relu")

        remove_btn = QPushButton("Видалити")
        row_layout.addWidget(spin)
        row_layout.addWidget(combo)
        row_layout.addWidget(remove_btn)

        container_layout = self.ui.add_layers_model_2.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(self.ui.add_layers_model_2)
            container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.addWidget(row_widget)

        layer_config = {"units": spin.value(), "activation": combo.currentText()}
        self.custom_layers.append(layer_config)

        spin.valueChanged.connect(lambda v, cfg=layer_config: cfg.update({"units": v}))
        combo.currentTextChanged.connect(lambda v, cfg=layer_config: cfg.update({"activation": v}))
        remove_btn.clicked.connect(lambda _, w=row_widget, cfg=layer_config: self.remove_layer(w, cfg))

    def remove_layer(self, widget, layer_config):
        try:
            if layer_config in self.custom_layers:
                self.custom_layers.remove(layer_config)
        except ValueError:
            pass
        widget.setParent(None)
        widget.deleteLater()

        if not self.custom_layers:
            QMessageBox.warning(self, "Попередження", "Має бути хоча б один шар у моделі!")
            self.add_layer()

    # ========================== НАВЧАННЯ КЛАСИФІКАЦІЇ ==========================
    def run_classification_model(self):
        if not self.custom_layers:
            QMessageBox.warning(self, "Помилка", "Модель має містити хоча б один шар!")
            return

        if hasattr(self.ui, "target_column_1") and self.ui.target_column_1.isVisible():
            self.target_column = self.ui.target_column_1.currentText()
        else:
            self.target_column = None

        if not self.target_column:
            QMessageBox.warning(self, "Помилка", "Оберіть цільову колонку (target_column_1)!")
            return

        # epochs
        epochs = 50
        if hasattr(self.ui, "epochs_size_1"):
            try:
                txt = self.ui.epochs_size_1.text().strip()
                epochs = int(txt) if txt else 50
            except Exception:
                epochs = 50

        # result_size
        result_size = 5
        if hasattr(self.ui, "result_size_1"):
            try:
                txt = self.ui.result_size_1.text().strip()
                result_size = int(txt) if txt else 5
            except Exception:
                result_size = 5

        # test_size
        test_size = 0.2
        if hasattr(self.ui, "test_size_1"):
            try:
                txt = self.ui.test_size_1.text().strip()
                if txt:
                    test_size = float(txt)
                if not (0.05 <= test_size <= 0.95):
                    raise ValueError
            except Exception:
                QMessageBox.warning(
                    self,
                    "Некоректне значення",
                    "Поле 'test_size_1' має бути числом від 0.1 до 0.9. Використано стандартне значення 0.2.",
                )
                test_size = 0.2

        self.worker = ClassificationWorker(
            self.data_path,
            target_column=self.target_column,
            epochs=epochs,
            test_size=test_size,
            layers=self.custom_layers,
            result_size=result_size
        )
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.show_result)
        self.worker.start()

        print(f"[UI] Запущено навчання класифікації: epochs={epochs}, test_size={test_size}, layers={self.custom_layers}")

    def update_status(self, message):
        print("[Status]", message)
        try:
            self.ui.result_model_2.setText(message)
        except Exception:
            pass

    def show_result(self, result_text, model):
        self.trained_model = model
        print("[Result]\n", result_text)
        try:
            self.ui.result_model_2.setText(result_text)
        except Exception:
            pass

    # ========================== ЗБЕРЕЖЕННЯ МОДЕЛІ ==========================
    def save_trained_model(self):
        if self.trained_model is None:
            QMessageBox.warning(self, "Немає моделі", "Спочатку потрібно навчити модель або завантажити її!")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти модель як", "", "Keras Model (*.keras);;H5 Model (*.h5)"
        )
        if not save_path:
            return

        try:
            self.trained_model.save(save_path)
            QMessageBox.information(self, "Успіх", f"Модель збережено у файл:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти модель:\n{e}")

    # ========================== ЗАВАНТАЖЕННЯ МОДЕЛІ (ТИ НАПИШЕШ ЛОГІКУ) ==========================
    def load_trained_model(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити модель", "", "Keras Model (*.keras *.h5)"
        )
        if not file_path:
            return

        try:
            self.trained_model = load_model(file_path)
            QMessageBox.information(self, "Успіх", f"Модель успішно завантажено:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити модель:\n{e}")
            return

        # === Перевірка наявності даних ===
        if not hasattr(self, "data_path") or not os.path.exists(self.data_path):
            self.ui.result_model_2.setText("Модель завантажена. Даних для прогнозу не знайдено.")
            return

        try:
            # === Завантаження даних ===
            if self.data_path.endswith(".xlsx"):
                data = pd.read_excel(self.data_path, engine="openpyxl")
            elif self.data_path.endswith(".xls"):
                data = pd.read_excel(self.data_path, engine="xlrd")
            else:
                data = pd.read_csv(self.data_path)

            target_col = self.ui.target_column_1.currentText().strip()
            if target_col not in data.columns:
                raise ValueError(f"У файлі немає колонки '{target_col}'")

            # === Відокремлення цільової колонки ===
            y = data[target_col]
            X = data.drop(columns=[target_col])

            # === Кодування нечислових ознак ===
            for col in X.columns:
                if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
                    encoder = LabelEncoder()
                    try:
                        X[col] = encoder.fit_transform(X[col].astype(str))
                    except Exception:
                        X[col] = pd.factorize(X[col].astype(str))[0]

            # === Масштабування ===
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # === Кодування цільових класів ===
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)

            # === Прогноз ===
            y_pred_probs = self.trained_model.predict(X_scaled, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            result_size = 4
            if hasattr(self.ui, "result_size_1"):
                try:
                    txt = self.ui.result_size_1.text().strip()
                    result_size = int(txt) if txt else 4
                except Exception:
                    result_size = 4

            result_size = min(result_size, len(y_encoded))

            preds = target_encoder.inverse_transform(y_pred[:result_size])
            reals = target_encoder.inverse_transform(y_encoded[:result_size])

            # === Обчислення точності ===
            accuracy = np.mean(y_pred == y_encoded) * 100

            # === Форматований результат ===
            result_text = (
                f"Завантажена модель: {os.path.basename(file_path)}\n"
                f"Цільова колонка: {target_col}\n"
                f"Точність (по всіх даних): {accuracy:.2f}%\n\n"
                f"Класи: {list(target_encoder.classes_)}\n\n"
                f"Прогнози (перші {result_size}): {preds}\n"
                f"Реальні значення: {reals}"
            )

            print("[Loaded classification model result]\n", result_text)
            self.ui.result_model_2.setText(result_text)

        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зробити прогноз:\n{e}")
