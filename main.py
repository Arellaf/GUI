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
        self.custom_layers = []  # список Dense-шарів

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

    # ========================== ВИБІР ФАЙЛУ ==========================
    def select_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть ваш Excel-файл", "", "Excel файли (*.xlsx *.xls)"
        )
        if not file_path:
            return
        self.data_path = file_path
        QMessageBox.information(self, "Файл вибрано", f"Використовується файл:\n{file_path}")

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

    # ========================== НАВЧАННЯ ==========================
    def run_regression_model(self):
        if not os.path.exists(self.data_path):
            QMessageBox.critical(self, "Помилка", "Файл для навчання не знайдено!")
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

        self.worker = RegressionWorker(
            self.data_path,
            epochs=epochs,
            test_size=0.2,
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
            self.trained_model = load_model(file_path)
            QMessageBox.information(self, "Успіх", f"Модель успішно завантажено:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити модель:\n{e}")
            return

        if os.path.exists(self.data_path):
            try:
                data = pd.read_excel(self.data_path, engine="openpyxl")
                X = data[["hours_studied", "attendance", "assignments_completed"]].values
                X = X / X.max(axis=0)
                y = data["final_score"].values

                result_size = 4
                if hasattr(self.ui, "result_size"):
                    try:
                        txt = self.ui.result_size.text().strip()
                        result_size = int(txt) if txt != "" else 4
                    except Exception:
                        result_size = 4
                result_size = max(1, min(result_size, len(X)))

                # Прогноз
                y_pred_all = self.trained_model.predict(X).flatten() * 100
                preds = y_pred_all[:result_size]
                real_values = y[:result_size]

                # ==== Обчислення точності у % ====
                mape = np.mean(np.abs((y - y_pred_all) / y)) * 100
                accuracy = 100 - mape

                out_text = (
                    f"Завантажена модель: {os.path.basename(file_path)}\n"
                    f"Прогноз (перші {result_size}): {np.array2string(preds, precision=2, floatmode='fixed', separator=', ')}\n"
                    f"Реальні значення: {np.array2string(real_values, precision=2, floatmode='fixed', separator=', ')}\n\n"
                    f"Точність моделі: {accuracy:.2f}%"
                )

                print("[Loaded model prediction]\n", out_text)
                try:
                    self.ui.result_model_1.setText(out_text)
                except Exception:
                    pass

            except Exception as e:
                QMessageBox.critical(self, "Помилка", f"Не вдалося зробити прогноз:\n{e}")
        else:
            try:
                self.ui.result_model_1.setText("Модель завантажена. Даних для прогнозу не знайдено.")
            except Exception:
                pass


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
