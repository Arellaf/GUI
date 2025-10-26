# main.py
import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QMessageBox,
    QSpinBox, QComboBox, QHBoxLayout, QVBoxLayout, QWidget, QPushButton
)
from PyQt5.QtCore import QThread
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

from main_UI import Ui_MainWindow
from workers.regression_worker import RegressionWorker  # переконайся, що файл є


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(BASE_DIR, "data", "student_scores.xlsx")
        self.trained_model = None
        self.custom_layers = []

        # --- Навігація ---
        try:
            self.ui.icon_widget.hide()
        except Exception:
            pass
        try:
            self.ui.content.setCurrentIndex(0)
        except Exception:
            pass
        try:
            self.ui.lab1_btn_2.setChecked(True)
        except Exception:
            pass

        # --- Кнопки сторінок ---
        try:
            self.ui.lab1_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(0))
            self.ui.lab1_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(0))
            self.ui.lab2_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(1))
            self.ui.lab2_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(1))
            self.ui.lab3_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(2))
            self.ui.lab3_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(2))
        except Exception:
            pass

        # --- Переходи ---
        try:
            self.ui.stackedWidget_btn_next_1.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
            self.ui.stackedWidget_btn_next_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
            self.ui.stackedWidget_btn_back_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        except Exception:
            pass

        # --- Основні кнопки ---
        try:
            self.ui.upload_my_data.clicked.connect(self.select_excel_file)
        except Exception:
            pass

        try:
            self.ui.learn_btn_1.clicked.connect(self.run_regression_model)
        except Exception:
            pass

        try:
            self.ui.safe_model_btn.clicked.connect(self.save_trained_model)
        except Exception:
            pass

        # Кнопка "Завантажити модель"
        if hasattr(self.ui, "upload_my_model"):
            try:
                self.ui.upload_my_model.clicked.connect(self.load_trained_model)
            except Exception:
                pass
        else:
            try:
                self.ui.upload_my_model = QPushButton("Завантажити модель", self)
                self.ui.upload_my_model.setFixedSize(160, 30)
                self.ui.upload_my_model.move(self.width() - 180, self.height() - 50)
                self.ui.upload_my_model.clicked.connect(self.load_trained_model)
            except Exception:
                pass

        # --- Додавання шарів ---
        if hasattr(self.ui, "add_layer_btn"):
            try:
                self.ui.add_layer_btn.clicked.connect(self.add_layer)
            except Exception:
                pass

        self.worker = None

        try:
            self.add_layer()
        except Exception as e:
            print("Не вдалося створити базовий шар:", e)

        self.ui.target_column.hide()

    # ========================== ВИБІР ФАЙЛУ ==========================
    def select_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть ваш Excel-файл", "", "Excel файли (*.xlsx *.xls)"
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

            if hasattr(self.ui, "target_column"):
                self.ui.target_column.clear()
                self.ui.target_column.addItems(columns)
                self.ui.target_column.setEnabled(True)
                self.ui.target_column.show()
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося прочитати файл:\n{e}")


    # ========================== КАСТОМНІ ШАРИ ==========================
    def add_layer(self):
        if not hasattr(self.ui, "add_layers_model_1"):
            QMessageBox.warning(self, "UI error", "Віджет add_layers_model_1 не знайдено у UI.")
            return

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 2, 2, 2)

        spin = QSpinBox()
        spin.setRange(1, 4096)
        spin.setValue(256)

        combo = QComboBox()
        combo.addItems(["relu", "sigmoid", "tanh", "linear"])
        combo.setCurrentText("relu")

        remove_btn = QPushButton("Видалити")
        row_layout.addWidget(spin)
        row_layout.addWidget(combo)
        row_layout.addWidget(remove_btn)

        container_layout = self.ui.add_layers_model_1.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(self.ui.add_layers_model_1)
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

    # ========================== НАВЧАННЯ ==========================
    def run_regression_model(self):
        if not self.custom_layers:
            QMessageBox.warning(self, "Помилка", "Модель має містити хоча б один шар!")
            return

        # --- Отримуємо target column ---
        if hasattr(self.ui, "target_column") and self.ui.target_column.isVisible():
            self.target_column = self.ui.target_column.currentText()
        else:
            self.target_column = None

        # --- Перевірка ---
        if not self.target_column:
            QMessageBox.warning(self, "Помилка", "Оберіть цільову колонку (target_column)!")
            return

        epochs = 50
        if hasattr(self.ui, "epochs_size"):
            try:
                txt = self.ui.epochs_size.text().strip()
                epochs = int(txt) if txt != "" else 50
            except Exception:
                epochs = 50

        epochs = max(1, epochs)

        result_size = 4
        if hasattr(self.ui, "result_size"):
            try:
                txt = self.ui.result_size.text().strip()
                result_size = int(txt) if txt != "" else 4
            except Exception:
                result_size = 4

        result_size = max(1, result_size)

        # ==== test_size ====
        test_size = 0.2
        if hasattr(self.ui, "test_size"):
            try:
                txt = self.ui.test_size.text().strip()
                if txt != "":
                    test_size = float(txt)
                if not (0.05 <= test_size <= 0.95):
                    raise ValueError
            except Exception:
                QMessageBox.warning(
                    self,
                    "Некоректне значення",
                    "Поле 'test_size' має бути числом від 0.1 до 0.9. Використано стандартне значення 0.2.",
                )
                test_size = 0.2

        self.worker = RegressionWorker(
            self.data_path,
            target_column=self.target_column,
            epochs=epochs,
            test_size=test_size,
            layers=self.custom_layers,
            result_size=result_size
        )
        try:
            self.worker.progress.connect(self.update_status)
            self.worker.finished.connect(self.show_result)
        except Exception:
            pass

        self.worker.start()
        print(f"[UI] Запущено навчання: epochs={epochs}, result_size={result_size}, layers={self.custom_layers}")

    def update_status(self, message):
        print("[Status]", message)
        try:
            self.ui.result_model_1.setText(message)
        except Exception:
            pass

    def show_result(self, result_text, model):
        self.trained_model = model
        print("[Result]\n", result_text)
        try:
            self.ui.result_model_1.setText(result_text)
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

    # ========================== ЗАВАНТАЖЕННЯ МОДЕЛІ ==========================
    def load_trained_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити модель", "", "Keras Model (*.keras *.h5)"
        )
        if not file_path:
            return

        try:
            from tensorflow.keras.models import load_model
            self.trained_model = load_model(file_path)
            QMessageBox.information(self, "Успіх", f"Модель успішно завантажено:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити модель:\n{e}")
            return

        # === Перевірка даних ===
        if not hasattr(self, "data_path") or not os.path.exists(self.data_path):
            try:
                self.ui.result_model_1.setText("Модель завантажена. Даних для прогнозу не знайдено.")
            except Exception:
                pass
            return

        try:
            # === Завантаження даних ===
            if self.data_path.endswith((".xlsx", ".xls")):
                data = pd.read_excel(self.data_path, engine="openpyxl")
            else:
                data = pd.read_csv(self.data_path)

            target_col = self.ui.target_column.currentText().strip()

            if target_col not in data.columns:
                raise ValueError(f"У файлі немає колонки '{target_col}'")

            X = data.drop(columns=[target_col]).select_dtypes(include=[np.number])
            y = data[target_col].values

            if X.shape[1] == 0:
                raise ValueError("У файлі немає числових ознак для прогнозу.")

            # === Нормалізація ===
            X = X / X.max(axis=0)
            y = y / 100.0

            # === Прогноз ===
            y_pred_all = self.trained_model.predict(X).flatten() * 100
            y_real_all = y * 100

            # === Кількість результатів для показу ===
            result_size = 4
            if hasattr(self.ui, "result_size"):
                try:
                    txt = self.ui.result_size.text().strip()
                    result_size = int(txt) if txt else 4
                except Exception:
                    result_size = 4
            result_size = min(result_size, len(X))

            predictions = y_pred_all[:result_size]
            real_values = y_real_all[:result_size]

            # === Обчислення точності ===
            mape = np.mean(np.abs((y_real_all - y_pred_all) / y_real_all)) * 100
            accuracy = 100 - mape

            # === Форматований текст ===
            result_text = (
                f"Завантажена модель: {os.path.basename(file_path)}\n"
                f"Цільова колонка: {target_col}\n"
                f"Втрати (MSE): немає (модель лише для прогнозу)\n"
                f"Точність моделі (приблизно): {accuracy:.2f}%\n\n"
                f"Прогнози (перші {result_size}): "
                f"{np.array2string(predictions, precision=3, floatmode='fixed', separator=', ')}\n"
                f"Реальні значення: "
                f"{np.array2string(real_values, precision=1, floatmode='fixed', separator=', ')}"
            )

            print("[Loaded model prediction]\n", result_text)
            try:
                self.ui.result_model_1.setText(result_text)
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зробити прогноз:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        with open("style.qss", "r") as f:
            qss = f.read()
            app.setStyleSheet(qss)
    except FileNotFoundError:
        print("Увага: файл style.qss не знайдено!")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
