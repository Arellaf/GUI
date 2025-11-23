# File: UI/model3_page.py (updated show_result to set pixmap on label_7)
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QPushButton, QSpinBox, QLabel
)
from workers.NeuroFuzzy_worker import NeuroFuzzyWorker

class Model3Page(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        if os.path.isdir(self.data_path):
            files = [f for f in os.listdir(self.data_path) if f.endswith((".csv", ".xlsx", ".xls"))]
            if files:
                self.data_path = os.path.join(self.data_path, files[0])
        else:
            self.data_path = ""

        self.trained_model = None
        self.custom_layers = []
        self.worker = None

        try:
            self.ui.upload_my_data_3.clicked.connect(self.select_file)
        except Exception:
            pass
        try:
            self.ui.learn_btn_4.clicked.connect(self.run_neurofuzzy_model)
        except Exception:
            pass
        try:
            self.ui.safe_model_btn_3.clicked.connect(self.save_trained_model)
        except Exception:
            pass
        try:
            self.ui.upload_my_model_3.clicked.connect(self.load_trained_model)
        except Exception:
            pass
        try:
            self.ui.add_layer_btn_3.clicked.connect(self.add_layer)
        except Exception:
            pass

        try:
            self.add_layer()
        except Exception as e:
            print("Не вдалось добавити початковий слой:", e)

        try:
            self.ui.target_column_3.hide()
        except Exception:
            pass

        try:
            self.ui.stackedWidget_btn_next_7.clicked.connect(lambda: self.ui.stackedWidget_3.setCurrentIndex(1))
            self.ui.stackedWidget_btn_next_8.clicked.connect(lambda: self.ui.stackedWidget_3.setCurrentIndex(2))
            self.ui.stackedWidget_btn_back_5.clicked.connect(lambda: self.ui.stackedWidget_3.setCurrentIndex(0))
        except Exception:
            pass

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл с данными", "", "CSV/Excel (*.csv *.xlsx *.xls)")
        if not file_path:
            return
        self.data_path = file_path
        QMessageBox.information(self, "Файл выбран", f"Используется файл:\n{file_path}")
        try:
            if file_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path, nrows=0, engine="openpyxl")
            else:
                df = pd.read_csv(file_path, nrows=0)
            columns = df.columns.tolist()
            if hasattr(self.ui, "target_column_3"):
                self.ui.target_column_3.clear()
                self.ui.target_column_3.addItems(columns)
                self.ui.target_column_3.setEnabled(True)
                self.ui.target_column_3.show()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл:\n{e}")

    def add_layer(self):
        if not hasattr(self.ui, "add_layers_model_4"):
            QMessageBox.warning(self, "UI error", "Виджет add_layers_model_4 не найден в UI.")
            return

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2,2,2,2)

        spin = QSpinBox()
        spin.setRange(1, 1024)
        spin.setValue(8)

        lbl = QLabel("output_dim:")
        remove_btn = QPushButton("Видалити")

        row_layout.addWidget(lbl)
        row_layout.addWidget(spin)
        row_layout.addWidget(remove_btn)

        container_layout = self.ui.add_layers_model_4.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(self.ui.add_layers_model_4)
            container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.addWidget(row_widget)

        layer_config = {"output_dim": spin.value()}
        self.custom_layers.append(layer_config)

        spin.valueChanged.connect(lambda v, cfg=layer_config: cfg.update({"output_dim": v}))
        remove_btn.clicked.connect(lambda _, w=row_widget, cfg=layer_config: self.remove_layer(w, cfg))

    def remove_layer(self, widget, layer_config):
        try:
            if layer_config in self.custom_layers:
                self.custom_layers.remove(layer_config)
        except Exception:
            pass
        widget.setParent(None)
        widget.deleteLater()

        if not self.custom_layers:
            QMessageBox.warning(self, "Внимание", "Должен быть хотя бы один слой. Добавлен базовый слой.")
            self.add_layer()

    def run_neurofuzzy_model(self):
        if not self.custom_layers:
            QMessageBox.warning(self, "Ошибка", "Модель должна содержать хотя бы один нечёткий слой!")
            return

        if hasattr(self.ui, "target_column_3") and self.ui.target_column_3.isVisible():
            self.target_column = self.ui.target_column_3.currentText()
        else:
            QMessageBox.warning(self, "Ошибка", "Выберите целевую колонку (target_column_3)!")
            return

        epochs = 50
        if hasattr(self.ui, "epochs_size_3"):
            try:
                txt = self.ui.epochs_size_3.text().strip()
                epochs = int(txt) if txt else 50
            except Exception:
                epochs = 50

        result_size = 4
        if hasattr(self.ui, "result_size_3"):
            try:
                txt = self.ui.result_size_3.text().strip()
                result_size = int(txt) if txt else 4
            except Exception:
                result_size = 4

        test_size = 0.2
        if hasattr(self.ui, "test_size_3"):
            try:
                txt = self.ui.test_size_3.text().strip()
                if txt:
                    test_size = float(txt)
                if not (0.05 <= test_size <= 0.95):
                    raise ValueError
            except Exception:
                QMessageBox.warning(self, "Некорректное значение", "Поле 'test_size_3' некорректно. Использовано 0.2")
                test_size = 0.2

        self.worker = NeuroFuzzyWorker(
            filepath=self.data_path,
            target_column=self.target_column,
            epochs=epochs,
            test_size=test_size,
            fuzzy_layers=self.custom_layers,
            result_size=result_size
        )
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.show_result)
        self.worker.start()

        print(f"[UI] Запуск Neuro-Fuzzy: epochs={epochs}, layers={self.custom_layers}, test_size={test_size}")

    def update_status(self, message):
        print("[Status]", message)
        try:
            self.ui.result_model_4.setText(message)
        except Exception:
            pass

    def show_result(self, result_text, model, y_real, y_pred, plot_path):
        self.trained_model = model
        print("[Result]\n", result_text)
        try:
            self.ui.result_model_4.setText(result_text)
        except Exception:
            pass

        try:
            if plot_path and hasattr(self.ui, "label_7"):
                from PyQt5.QtGui import QPixmap
                pix = QPixmap(plot_path)
                self.ui.label_7.setPixmap(pix)
            elif plot_path:
                try:
                    self.ui.result_model_4.setText(self.ui.result_model_4.text() + f"\n\nГрафик: {plot_path}")
                except Exception:
                    pass
        except Exception:
            pass

    def save_trained_model(self):
        if self.trained_model is None:
            QMessageBox.warning(self, "Нет модели", "Сначала обучите модель или загрузите её!")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель как", "", "Keras Model (*.keras);;H5 Model (*.h5)")
        if not save_path:
            return
        try:
            self.trained_model.save(save_path)
            QMessageBox.information(self, "Успіх", f"Модель сохранена: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не удалось сохранить модель:\n{e}")

    def load_trained_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить модель", "", "Keras Model (*.keras *.h5)")
        if not file_path:
            return
        try:
            from tensorflow.keras.models import load_model
            # pass both custom layers
            import layers.FuzzyLayer as FL
            import layers.DefuzzyLayer as DL
            custom_objects = {"FuzzyLayer": FL.FuzzyLayer, "DefuzzyLayer": DL.DefuzzyLayer}
            self.trained_model = load_model(file_path, custom_objects=custom_objects)
            QMessageBox.information(self, "Успіх", f"Модель загружена: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалось загрузити модель:\n{e}")
            return

        if not hasattr(self, "data_path") or not os.path.exists(self.data_path):
            try:
                self.ui.result_model_4.setText("Модель загружена. Данные для прогноза не найдены.")
            except Exception:
                pass
            return

        try:
            if self.data_path.endswith((".xlsx", ".xls")):
                data = pd.read_excel(self.data_path, engine="openpyxl")
            else:
                data = pd.read_csv(self.data_path)

            target_col = self.ui.target_column_3.currentText().strip()
            if target_col not in data.columns:
                raise ValueError(f"В файле нет колонки '{target_col}'")

            X = data.drop(columns=[target_col]).select_dtypes(include=[np.number])
            y = data[target_col].values

            if X.shape[1] == 0:
                raise ValueError("В файле нет числовых признаков для прогноза.")

            X_norm = X / (X.max(axis=0).replace(0, 1))
            y_norm = y / 100.0

            y_pred_all = self.trained_model.predict(X_norm.values).flatten() * 100
            y_real_all = y_norm * 100

            result_size = 4
            if hasattr(self.ui, "result_size_3"):
                try:
                    txt = self.ui.result_size_3.text().strip()
                    result_size = int(txt) if txt else 4
                except Exception:
                    result_size = 4
            result_size = min(result_size, len(X))

            predictions = y_pred_all[:result_size]
            real_values = y_real_all[:result_size]

            mask = y_real_all != 0
            if mask.sum() == 0:
                mape = float("nan")
                accuracy = float("nan")
            else:
                mape = np.mean(np.abs((y_real_all[mask] - y_pred_all[mask]) / y_real_all[mask])) * 100
                accuracy = 100 - mape

            result_text = (
                f"Загружена модель: {os.path.basename(file_path)}\n"
                f"Целевая колонка: {target_col}\n"
                f"Точность (примерно): {accuracy:.2f}%\n\n"
                f"Прогнозы (первые {result_size}): "
                f"{np.array2string(predictions, precision=3, floatmode='fixed', separator=', ')}\n"
                f"Реальные значения: "
                f"{np.array2string(real_values[:result_size], precision=1, floatmode='fixed', separator=', ')}"
            )

            try:
                self.ui.result_model_4.setText(result_text)
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не вдалось зробити прогноз:\n{e}")
