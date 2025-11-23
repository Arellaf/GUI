# File: UI/model3_page.py (updated show_result to set pixmap on label_7)
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QPushButton, QSpinBox, QLabel, QComboBox
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
            print("Не вдалося додати початковий шар:", e)

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Виберіть файл із даними", "", "CSV/Excel (*.csv *.xlsx *.xls)")
        if not file_path:
            return
        self.data_path = file_path
        QMessageBox.information(self, "Файл вибраний", f"Використовується файл:\n{file_path}")
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
            QMessageBox.critical(self, "Помилка", f"Неможливо прочитати файл:\n{e}")

    def add_layer(self):
        if not hasattr(self.ui, "add_layers_model_4"):
            QMessageBox.warning(self, "UI error", "Віджет add_layers_model_4 не знайдено")
            return

        # === Контейнер для шарів ===
        container_layout = self.ui.add_layers_model_4.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(self.ui.add_layers_model_4)
            container_layout.setContentsMargins(4, 4, 4, 4)

        # Якщо це перший шар — створюємо FuzzyLayer
        if len(self.custom_layers) == 0:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(2, 2, 2, 2)

            lbl_fz = QLabel("FuzzyLayer output_dim:")
            spin_fz = QSpinBox()
            spin_fz.setRange(1, 128)
            spin_fz.setValue(8)

            row_layout.addWidget(lbl_fz)
            row_layout.addWidget(spin_fz)

            # Додаємо в контейнер
            container_layout.addWidget(row_widget)

            layer_config = {"type": "fuzzy", "output_dim": spin_fz.value()}
            self.custom_layers.append(layer_config)

            spin_fz.valueChanged.connect(lambda v, cfg=layer_config: cfg.update({"output_dim": v}))

            return

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 2, 2, 2)

        # Units
        spin_units = QSpinBox()
        spin_units.setRange(1, 2048)
        spin_units.setValue(128)

        lbl_units = QLabel("units:")

        # Activation
        combo_act = QComboBox()
        combo_act.addItems(["relu", "sigmoid", "tanh", "linear"])

        lbl_act = QLabel("activation:")

        # Remove button
        remove_btn = QPushButton("Видалити")

        # Add widgets
        row_layout.addWidget(lbl_units)
        row_layout.addWidget(spin_units)
        row_layout.addWidget(lbl_act)
        row_layout.addWidget(combo_act)
        row_layout.addWidget(remove_btn)

        container_layout.addWidget(row_widget)

        # Store config
        layer_config = {
            "type": "dense",
            "units": spin_units.value(),
            "activation": combo_act.currentText()
        }
        self.custom_layers.append(layer_config)

        spin_units.valueChanged.connect(lambda v, cfg=layer_config: cfg.update({"units": v}))
        combo_act.currentTextChanged.connect(lambda v, cfg=layer_config: cfg.update({"activation": v}))

        remove_btn.clicked.connect(lambda _, w=row_widget, cfg=layer_config: self.remove_layer(w, cfg))

    def remove_layer(self, widget, layer_config):
        if layer_config.get("type") == "fuzzy":
            QMessageBox.warning(self, "Помилка", "Перший FuzzyLayer не можна видалити!")
            return

        try:
            if layer_config in self.custom_layers:
                self.custom_layers.remove(layer_config)
        except:
            pass

        widget.setParent(None)
        widget.deleteLater()

    def run_neurofuzzy_model(self):
        if not self.custom_layers:
            QMessageBox.warning(self, "Помилка", "Модель повинна містити хоча б один нечіткий шар!")
            return

        if hasattr(self.ui, "target_column_3") and self.ui.target_column_3.isVisible():
            self.target_column = self.ui.target_column_3.currentText()
        else:
            QMessageBox.warning(self, "Помилка", "Виберіть цільову колонку!")
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
                QMessageBox.warning(self, "Некоректне значення", "Поле 'test_size_3' некоректно. Використано 0.2")
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
            QMessageBox.warning(self, "Немає моделі", "Спочатку навчіть модель або завантажте її!")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Зберегти модель як", "", "Keras Model (*.keras);;H5 Model (*.h5)")
        if not save_path:
            return
        try:
            self.trained_model.save(save_path)
            QMessageBox.information(self, "Успіх", f"Модель збережена: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти модель:\n{e}")

    def load_trained_model(self):
        from tensorflow.keras.models import load_model
        from layers.FuzzyLayer import FuzzyLayer
        from layers.DefuzzyLayer import DefuzzyLayer

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити модель", "", "Keras Model (*.keras *.h5)"
        )
        if not file_path:
            return

        try:
            self.trained_model = load_model(
                file_path,
                custom_objects={
                    "FuzzyLayer": FuzzyLayer,
                    "DefuzzyLayer": DefuzzyLayer
                }
            )

            QMessageBox.information(self, "Успіх",
                                    f"Модель успішно завантажена:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Помилка",
                                 f"Не вдалося завантажити модель:\n{e}")
            return

        # --- Якщо даних для прогнозу нема ---
        if not hasattr(self, "data_path") or not os.path.exists(self.data_path):
            try:
                self.ui.result_model_4.setText("Модель завантажена, але дані для прогнозу відсутні.")
            except:
                pass
            return

        # --- Спроба виконати прогноз ---
        try:
            # читання даних
            if self.data_path.endswith((".xlsx", ".xls")):
                data = pd.read_excel(self.data_path, engine="openpyxl")
            else:
                data = pd.read_csv(self.data_path)

            target_col = self.ui.target_column_3.currentText().strip()
            if target_col not in data.columns:
                raise ValueError(f"У файлі немає цільової колонки '{target_col}'")

            X = data.drop(columns=[target_col]).select_dtypes(include=[np.number])
            y = data[target_col].values.astype(float)

            if X.shape[1] == 0:
                raise ValueError("У файлі немає числових ознак для прогнозу.")

            # нормалізація (твоя модель використовує /100)
            X_norm = X / (X.max(axis=0).replace(0, 1))
            y_real_all = y.copy()

            # прогноз
            y_pred_all = self.trained_model.predict(X_norm.values).flatten() * 100

            # кількість значень для відображення
            result_size = 4
            if hasattr(self.ui, "result_size_3"):
                try:
                    val = self.ui.result_size_3.text().strip()
                    result_size = int(val) if val else 4
                except:
                    pass

            result_size = min(result_size, len(y_pred_all))

            predictions = y_pred_all[:result_size]
            real_values = y_real_all[:result_size]

            # обчислення MAPE
            mask = real_values != 0
            if mask.sum() == 0:
                mape = float("nan")
                accuracy = float("nan")
            else:
                mape = np.mean(np.abs((real_values[mask] - predictions[mask]) / real_values[mask])) * 100
                accuracy = 100 - mape

            # результат
            result_text = (
                f"Модель: {os.path.basename(file_path)}\n"
                f"Цільова колонка: {target_col}\n"
                f"Приблизна точність: {accuracy:.2f}%\n\n"
                f"Прогнози: {np.array2string(predictions, precision=3, separator=', ')}\n"
                f"Реальні: {np.array2string(real_values, precision=1, separator=', ')}"
            )

            try:
                self.ui.result_model_4.setText(result_text)
            except:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося виконати прогноз:\n{e}")
