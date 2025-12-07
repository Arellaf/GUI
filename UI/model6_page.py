import os
import tensorflow as tf
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QWidget, QFileDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox
from sklearn.preprocessing import StandardScaler
import h5py
from sklearn.metrics import r2_score
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from workers.DNNTrainWorker import DNNTrainWorker
from UI.layer_widget import LayerWidget

class Model6Page(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui

        self.selected_file = None
        self.training_thread = None
        self.trained_model = None
        self.X_data = None
        self.target_column_name = None
        self.layer_widgets = []

        self.layers_layout = QVBoxLayout()
        self.ui.model_6_layers_container.setLayout(self.layers_layout)

        self.add_layer_btn = QPushButton("Add layer")
        self.add_layer_btn.clicked.connect(self.add_layer)
        self.layers_layout.addWidget(self.add_layer_btn)

        self.ui.model_6_start_learning.clicked.connect(self.start_learning)
        self.ui.model_6_upload_data.clicked.connect(self.select_file)
        self.ui.model_6_save.clicked.connect(self.save_model)
        self.ui.model_6_upload_model.clicked.connect(self.load_model)
        self.ui.model_6_reset.clicked.connect(self.reset_model)

        self.ui.stackedWidget_btn_next_4.clicked.connect(lambda: self.ui.model_6_container.setCurrentIndex(1))
        self.ui.stackedWidget_btn_back_3.clicked.connect(lambda: self.ui.model_6_container.setCurrentIndex(0))
        self.ui.stackedWidget_btn_next_6.clicked.connect(lambda: self.ui.model_6_container.setCurrentIndex(2))

        self.add_layer()

        self.ui.model_6_save.setEnabled(False)
        self.ui.model_6_progress.setValue(0)
        self.ui.model_6_progress.setFormat("Learning: %p%")
        self.ui.model_6_progress.hide()
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)

        layout = self.ui.model_6_graphic.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.ui.model_6_graphic.setLayout(layout)

        layout.addWidget(self.canvas)

        self.ui.model_6_target_column.hide()

        self.add_layer()

    def update_plot(self, history):
        """Оновлює графік у вбудованому полотні."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(history.history["loss"], label="MAE на навчанні")
        if "val_loss" in history.history:
            ax.plot(history.history["val_loss"], label="MAE на валідації")

        ax.set_title("Історія навчання")
        ax.set_xlabel("епохи")
        ax.set_ylabel("Втрати (MAE)")
        ax.grid(True)
        ax.legend()

        self.canvas.draw()

    def add_layer(self):
        widget = LayerWidget(len(self.layer_widgets), self.remove_layer)
        self.layer_widgets.append(widget)
        self.layers_layout.insertWidget(len(self.layer_widgets), widget)

    def remove_layer(self, widget):
        self.layer_widgets.remove(widget)
        widget.setParent(None)
        for i, w in enumerate(self.layer_widgets):
            w.label.setText(f"Layer {i + 1}:")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel file for training", "",
            "Excel Files (*.xls *.xlsx *.csv);;All Files (*)"
        )
        if not file_path:
            return

        self.selected_file = file_path
        self.ui.model_6_file.setText(f"Selected file: {os.path.basename(file_path)}")

        try:
            if file_path.endswith((".xls", ".xlsx")):
                df_preview = pd.read_excel(file_path, nrows=5)
            else:
                df_preview = pd.read_csv(file_path, nrows=5)

            numeric_cols = df_preview.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                self.show_error("У файлі немає числових колонок для цілі.")
                return

            self.ui.model_6_target_column.clear()
            self.ui.model_6_target_column.addItems(numeric_cols)
            self.ui.model_6_target_column.setEnabled(True)
            self.ui.model_6_target_column.show()

        except Exception as e:
            self.show_error(f"Помилка читання файлу: {e}")

    def start_learning(self):
        if not self.selected_file:
            self.show_error("Please select a file first.")
            return

        target_column = self.ui.model_6_target_column.currentText()
        if not target_column:
            self.show_error("Please select a target column.")
            return

        try:
            epochs = int(self.ui.model_6_epochs_input.text() or "30")
            dropout = float(self.ui.model_6_dropout_input.text() or "0.2")
            test_size = float(self.ui.model_6_test_size.text() or "0.2")
        except ValueError:
            self.show_error("Invalid numeric value")
            return

        layers_config = [(w.neurons.value(), w.activation.currentText()) for w in self.layer_widgets]

        self.target_column_name = target_column

        self.ui.model_6_progress.setValue(0)
        self.ui.model_6_progress.show()
        self.ui.model_6_start_learning.setEnabled(False)
        self.ui.model_6_start_learning.setText("Training...")

        try:
            self.training_thread = DNNTrainWorker(
                file_path=self.selected_file,
                target_column=target_column,
                epochs=epochs,
                layers=layers_config,
                dropout=dropout,
                test_size=test_size
            )

            self.training_thread.progress.connect(self.ui.model_6_progress.setValue)
            self.training_thread.message.connect(self.show_error)
            self.training_thread.finished.connect(self.training_done)

            self.training_thread.start()

        except Exception as e:
            self.show_error(f"Training start error: {str(e)}")

    def training_done(self, model, mae, history, x_test, y_test, y_pred, X, r2, scaler_X, scaler_y, _):
        self.ui.model_6_progress.hide()
        self.trained_model = model
        self.X_data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])  # або збережи колонки

        self.training_thread.scaler_X = scaler_X
        self.training_thread.scaler_y = scaler_y

        self.ui.model_6_save.setEnabled(True)
        self.ui.model_6_start_learning.setEnabled(True)
        self.ui.model_6_start_learning.setText("Start learning")
        self.ui.model_6_container.setCurrentIndex(1)

        self.show_predictions(mae, history, y_test, y_pred, r2)
        self.update_plot(history)

    def show_predictions(self, mae=None, history=None, y_test=None, y_pred=None, r2=None):
        if not self.trained_model or y_test is None or y_pred is None:
            self.show_error("Model is not trained or data is missing.")
            return

        n = self.ui.model_6_prediction_count.value()
        n = min(n, len(y_pred))
        sample_predictions = "\n".join([f"{i + 1}: Real: {y_test[i]:.4f}, Predicted: {y_pred[i]:.4f}"
                                      for i in range(n)])

        info = f"Target column: {getattr(self, 'target_column_name', 'Unknown')}\n"
        if mae is not None and history is not None and r2 is not None:
            info += (
                f"Model trained.\n"
                f"Test MAE: {mae:.4f}\n"
                f"R² Score: {r2:.4f}\n"
                f"Epochs: {len(history.history['loss'])}\n"
                f"Number of layers: {len(self.layer_widgets)}\n\n"
            )

        info += f"Predictions for '{getattr(self, 'target_column_name', 'Unknown')}' (first {n} rows):\n{sample_predictions}"
        self.ui.model_6_result.setText(info)

    def save_model(self):
        if not self.trained_model:
            return QMessageBox.critical(self, "Помилка", "Немає навченою моделі")
        if not hasattr(self, 'scaler_X') or self.scaler_X is None:
            if not hasattr(self.training_thread, 'scaler_X') or self.training_thread.scaler_X is None:
                return QMessageBox.critical(self, "Помилка", "Скалери не знайдені. Навчіть модель ще раз.")
            self.scaler_X = self.training_thread.scaler_X
            self.scaler_y = self.training_thread.scaler_y

        file_path, _ = QFileDialog.getSaveFileName(self, "Зберегти модель", "", "H5 Files (*.h5)")
        if not file_path: return
        if not file_path.endswith(".h5"): file_path += ".h5"

        try:
            n_features = self.scaler_X.n_features_in_
            if n_features is None:
                raise ValueError("Не вдалося визначити кількість ознак")

            new_model = tf.keras.Sequential()
            new_model.add(tf.keras.Input(shape=(n_features,)))

            old_norm = self.trained_model.layers[0]
            if old_norm.__class__.__name__ == "Normalization" and getattr(old_norm, 'adapted', False):
                new_norm = tf.keras.layers.Normalization(axis=-1)
                new_norm.adapt(np.zeros((1, n_features)))
                new_norm.set_weights(old_norm.get_weights())
                new_model.add(new_norm)
            else:
                new_model.add(tf.keras.layers.Normalization(axis=-1))

            for layer in self.trained_model.layers[1:]:
                new_model.add(layer)

            new_model.compile(optimizer='adam', loss='mae')

            new_model.save(file_path, save_format='h5')

            with h5py.File(file_path, 'a') as f:
                for key in ['scaler_X_mean', 'scaler_X_scale', 'scaler_y_mean', 'scaler_y_scale', 'target_column']:
                    if key in f: del f[key]

                f.create_dataset('scaler_X_mean', data=self.scaler_X.mean_)
                f.create_dataset('scaler_X_scale', data=self.scaler_X.scale_)
                f.create_dataset('scaler_y_mean', data=self.scaler_y.mean_.flatten())
                f.create_dataset('scaler_y_scale', data=self.scaler_y.scale_.flatten())
                f.attrs['target_column'] = self.target_column_name or "Unknown"

            QMessageBox.information(self, "Готово", f"Модель успішно збережена!\n{os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "Помилка збереження", str(e))
            import traceback, sys
            traceback.print_exc()

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Завантажити модель", "", "H5 Files (*.h5)")
        if not file_path:
            return

        try:
            from tensorflow.keras.models import load_model
            model = load_model(file_path, compile=False)
            model.compile(optimizer='adam', loss='mae')
            self.trained_model = model

            with h5py.File(file_path, 'r') as f:
                scaler_X = StandardScaler()
                scaler_X.mean_ = np.array(f['scaler_X_mean'])
                scaler_X.scale_ = np.array(f['scaler_X_scale'])
                scaler_X.var_ = scaler_X.scale_ ** 2
                scaler_X.n_features_in_ = len(scaler_X.mean_)

                scaler_y = StandardScaler()
                scaler_y.mean_ = np.array(f['scaler_y_mean']).reshape(-1, 1)
                scaler_y.scale_ = np.array(f['scaler_y_scale']).reshape(-1, 1)
                scaler_y.var_ = scaler_y.scale_ ** 2

                self.target_column_name = f.attrs.get('target_column', 'Unknown')
                if isinstance(self.target_column_name, bytes):
                    self.target_column_name = self.target_column_name.decode()

            self.scaler_X = scaler_X
            self.scaler_y = scaler_y

            self.ui.model_6_save.setEnabled(True)
            self.ui.model_6_container.setCurrentIndex(1)
            self.ui.model_6_result.setText(f"Модель завантажена: {os.path.basename(file_path)}")
            QMessageBox.information(self, "Готово", "Успішно!")

            if self.selected_file:
                self._predict_with_loaded_model()

        except Exception as e:
            QMessageBox.critical(self, "Помилка", str(e))

    def _predict_with_loaded_model(self):
        if not self.selected_file or not self.trained_model:
            return

        try:
            df = pd.read_excel(self.selected_file) if self.selected_file.endswith((".xls", ".xlsx")) else pd.read_csv(
                self.selected_file)
            df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

            target_col = self.target_column_name
            if target_col not in df.columns:
                self.ui.model_6_result.append(f"\nПопередження: колонка '{target_col}' відсутня у даних.")
                return

            X_raw = df.drop(columns=[target_col], errors='ignore').copy()

            for col in X_raw.columns:
                if X_raw[col].dtype == 'object':
                    X_raw[col] = X_raw[col].astype(str).str.strip().str.lower()
                    X_raw[col] = pd.factorize(X_raw[col])[0] + 1

            if X_raw.shape[1] != self.scaler_X.n_features_in_:
                self.ui.model_6_result.append(
                    f"\nПомилка: очікувалось {self.scaler_X.n_features_in_} ознак, отримано {X_raw.shape[1]}")
                return

            X = X_raw.values.astype(np.float32)
            X_scaled = self.scaler_X.transform(X)
            y_real = df[target_col].values.astype(np.float32)

            y_pred_scaled = self.trained_model.predict(X_scaled, verbose=0)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()

            n = min(self.ui.model_6_prediction_count.value(), len(y_pred))
            mae = np.mean(np.abs(y_real[:n] - y_pred[:n]))
            r2 = r2_score(y_real, y_pred)

            sample = "\n".join([f"{i + 1}: {y_real[i]:.3f} → {y_pred[i]:.3f}" for i in range(n)])

            self.ui.model_6_result.setText(
                f"Модель: {os.path.basename(self.selected_file)}\n"
                f"Ціль: {target_col}\n"
                f"MAE (перші {n}): {mae:.4f} | R²: {r2:.4f}\n\n"
                f"Прогнози:\n{sample}"
            )

        except Exception as e:
            self.ui.model_6_result.append(f"\nПомилка прогнозу: {e}")

    def reset_model(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.requestInterruption()
        self.training_thread = None
        self.trained_model = None
        self.X_data = None
        self.selected_file = None
        self.target_column_name = None

        self.ui.model_6_result.clear()
        self.ui.model_6_file.setText("")
        self.ui.model_6_save.setEnabled(False)
        self.ui.model_6_start_learning.setEnabled(True)
        self.ui.model_6_start_learning.setText("Start learning")
        self.ui.model_6_container.setCurrentIndex(0)

        for w in self.layer_widgets:
            w.setParent(None)
        self.layer_widgets.clear()
        self.add_layer()

        self.ui.model_6_progress.setValue(0)
        self.ui.model_6_progress.hide()

        self.ui.model_6_epochs_input.clear()
        self.ui.model_6_dropout_input.clear()
        self.ui.model_6_prediction_count.setValue(40)
        self.ui.model_6_target_column.hide()
        self.ui.model_6_target_column.clear()

    def show_error(self, message):
        self.reset_model()
        QMessageBox.critical(self, "Error", message)