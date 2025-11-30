import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QVBoxLayout, QPushButton,
    QMessageBox, QDialog, QVBoxLayout as QVLayout, QLabel
)
from PyQt5.QtCore import pyqtSignal, Qt
from sklearn.preprocessing import StandardScaler
import h5py
import tensorflow as tf
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from workers.GeneticTrainWorker import GeneticTrainWorker
from UI.layer_widget import LayerWidget


class PlotDialog(QDialog):
    def __init__(self, history, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Найкраще MSE для кожного покоління")
        self.setMinimumSize(640, 420)
        layout = QVLayout()
        self.setLayout(layout)

        if history is None or len(history) == 0:
            layout.addWidget(QLabel("Немає даних для побудови графіка."))
            return

        fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        gens = list(range(len(history)))
        ax.plot(gens, history, marker='o', linewidth=1)
        ax.set_xlabel("Покоління")
        ax.set_ylabel("Найкращий MSE")
        ax.set_title("Навчання GA: найкраща MSE за покоління")
        ax.grid(True)

        layout.addWidget(self.canvas)


class Model5Page(QWidget):
    compute_predictions_signal = pyqtSignal(object, float, object, object, object, object)

    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.selected_file = None
        self.training_thread = None
        self.trained_model = None
        self.X_data = None
        self.target_column_name = None
        self.layer_widgets = []

        # Налаштування шарів
        self.layers_layout = QVBoxLayout()
        self.ui.model_5_layers_container.setLayout(self.layers_layout)

        self.add_layer_btn = QPushButton("Add layer")
        self.add_layer_btn.clicked.connect(self.add_layer)
        self.layers_layout.addWidget(self.add_layer_btn)

        # Підключення кнопок
        self.ui.model_5_upload_data.clicked.connect(self.select_file)
        self.ui.model_5_start_learning.clicked.connect(self.start_learning)
        self.ui.model_5_save.clicked.connect(self.save_model)
        self.ui.model_5_upload_model.clicked.connect(self.load_model)
        self.ui.model_5_reset.clicked.connect(self.reset_model)

        # Стан UI
        self.ui.model_5_save.setEnabled(False)
        self.ui.model_5_progress.setValue(0)
        self.ui.model_5_progress.setFormat("Learning: %p%")
        self.ui.model_5_progress.hide()
        self.ui.model_5_target_column.hide()

        self.ui.stackedWidget_btn_next_9.clicked.connect(lambda: self.ui.model_5_container.setCurrentIndex(1))
        self.ui.stackedWidget_btn_back_6.clicked.connect(lambda: self.ui.model_5_container.setCurrentIndex(0))
        self.ui.stackedWidget_btn_next_10.clicked.connect(lambda: self.ui.model_5_container.setCurrentIndex(2))

        self.add_layer()
        self.add_layer()
        self.add_layer()

        self.compute_predictions_signal.connect(self.compute_predictions)

        print("[INFO] Model5Page initialized")

    def add_layer(self):
        widget = LayerWidget(len(self.layer_widgets), self.remove_layer)
        self.layer_widgets.append(widget)
        self.layers_layout.insertWidget(len(self.layer_widgets), widget)
        print(f"[INFO] Added layer {len(self.layer_widgets)}")

    def remove_layer(self, widget):
        self.layer_widgets.remove(widget)
        widget.setParent(None)
        for i, w in enumerate(self.layer_widgets):
            w.label.setText(f"Layer {i + 1}:")

    def select_file(self):
        print("[INFO] Opening file dialog to select training file")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel/CSV file for training", "",
            "Excel Files (*.xls *.xlsx *.csv);;All Files (*)"
        )
        if not file_path:
            return
        self.selected_file = file_path
        self.ui.model_5_file.setText(f"Selected file: {os.path.basename(file_path)}")
        print(f"[INFO] Selected file: {file_path}")

        try:
            df_preview = pd.read_excel(file_path, nrows=5) if file_path.endswith((".xls", ".xlsx")) else pd.read_csv(file_path, nrows=5)
            numeric_cols = df_preview.select_dtypes(include=[np.number]).columns.tolist()
            print(f"[INFO] Numeric columns detected: {numeric_cols}")
            if not numeric_cols:
                self.show_error("У файлі немає числових колонок для цілі.")
                return

            self.ui.model_5_target_column.clear()
            self.ui.model_5_target_column.addItems(numeric_cols)
            self.ui.model_5_target_column.setEnabled(True)
            self.ui.model_5_target_column.show()
        except Exception as e:
            self.show_error(f"Помилка читання файлу: {e}")

    def start_learning(self):
        if not self.selected_file:
            self.show_error("Спочатку виберіть файл.")
            return

        target_column = self.ui.model_5_target_column.currentText()
        if not target_column:
            self.show_error("Виберіть цільову колонку.")
            return

        try:
            population_size = int(self.ui.model_5_population_input.text() or "30")
            generations = int(self.ui.model_5_generations_input.text() or "100")
            mutation_percent = float(self.ui.model_5_mutation_input.text() or "2.0")
            adam_epochs = int(self.ui.model_5_epochs_input.text() or "30")
        except ValueError as e:
            self.show_error("Невірні параметри: перевірте числа.")
            return

        layers_config = [(w.neurons.value(), w.activation.currentText()) for w in self.layer_widgets]
        if not layers_config:
            self.show_error("Додайте хоча б один шар.")
            return

        self.target_column_name = target_column
        self.ui.model_5_progress.show()
        self.ui.model_5_progress.setValue(0)
        self.ui.model_5_start_learning.setEnabled(False)
        self.ui.model_5_start_learning.setText("Training...")

        print("[INFO] Starting training process")

        self.training_thread = GeneticTrainWorker(
            file_path=self.selected_file,
            target_column=target_column,
            population_size=population_size,
            generations=generations,
            mutation_percent=mutation_percent,
            layers=layers_config,
            adam_epochs=adam_epochs
        )

        self.training_thread.progress.connect(self.ui.model_5_progress.setValue)
        self.training_thread.message.connect(self.show_error)
        self.training_thread.finished.connect(self.compute_predictions_signal.emit)
        self.training_thread.start()

    def compute_predictions(self, weights_data, mse, ga, scaler_X, scaler_y, X_scaled):
        print("[INFO] Відновлення моделі в головному потоці...")
        self.ui.model_5_progress.hide()

        layers_config = [(w.neurons.value(), w.activation.currentText()) for w in self.layer_widgets]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_scaled.shape[1],)))
        for n, act in layers_config:
            model.add(tf.keras.layers.Dense(n, activation=act))
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        model.compile(optimizer='adam', loss='mse')

        weights = [np.array(w) for w in weights_data]
        model.set_weights(weights)

        self.trained_model = model
        self.X_data = pd.DataFrame(X_scaled, columns=[f"feature_{i}" for i in range(X_scaled.shape[1])])
        self.training_thread.scaler_X = scaler_X
        self.training_thread.scaler_y = scaler_y

        # Додаємо реальні значення
        df = pd.read_excel(self.selected_file) if self.selected_file.endswith((".xls", ".xlsx")) else pd.read_csv(
            self.selected_file)
        y_real = df[self.target_column_name].values
        y_real_scaled = scaler_y.transform(y_real.reshape(-1, 1)).flatten()

        self.ui.model_5_save.setEnabled(True)
        self.ui.model_5_start_learning.setEnabled(True)
        self.ui.model_5_start_learning.setText("Start learning")
        self.ui.model_5_container.setCurrentIndex(1)

        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

        self.show_predictions(y_test=y_real, y_pred=y_pred)

        try:
            history = ga
            if history is not None:
                dlg = PlotDialog(history, parent=self)
                dlg.exec_()
        except Exception as e:
            print(f"[WARN] Не вдалося побудувати графік: {e}")

    def show_predictions(self, y_test=None, y_pred=None):
        if self.trained_model is None or y_pred is None:
            self.show_error("Модель не навчена.")
            return

        n = min(self.ui.model_5_prediction_count.value(), len(y_pred))
        sample = "\n".join([f"{i + 1}: Real = {y_test[i]:.4f}, Predicted = {y_pred[i]:.4f}" for i in range(n)])

        info = f"Цільова колонка: {self.target_column_name}\n"
        info += f"Модель навчена успішно!\nMSE ≈ {getattr(self.training_thread, 'last_mse', '?.??????')}\n\n"
        info += f"Передбачення (перші {n} рядків з реальними значеннями):\n{sample}"

        self.ui.model_5_result.setText(info)

    def save_model(self):
        if not self.trained_model:
            QMessageBox.critical(self, "Помилка", "Немає навченої моделі.")
            return
        if not hasattr(self.training_thread, 'scaler_X'):
            QMessageBox.critical(self, "Помилка", "Скалери не збережені. Навчіть заново.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Зберегти модель", "", "H5 Files (*.h5)")
        if not file_path:
            return
        if not file_path.endswith(".h5"):
            file_path += ".h5"

        try:
            self.trained_model.save(file_path, save_format='h5')
            with h5py.File(file_path, 'a') as f:
                for name in ['scaler_X_mean', 'scaler_X_scale', 'scaler_y_mean', 'scaler_y_scale']:
                    if name in f:
                        del f[name]
                f.create_dataset('scaler_X_mean', data=self.training_thread.scaler_X.mean_)
                f.create_dataset('scaler_X_scale', data=self.training_thread.scaler_X.scale_)
                f.create_dataset('scaler_y_mean', data=self.training_thread.scaler_y.mean_.flatten())
                f.create_dataset('scaler_y_scale', data=self.training_thread.scaler_y.scale_.flatten())
                f.attrs['target_column'] = self.target_column_name or "Unknown"
            QMessageBox.information(self, "Готово", f"Модель збережена:\n{file_path}")
            self.ui.model_5_result.setText(f"Збережено:\n{os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти:\n{e}")

    def predict_after_load(self):
        if not self.selected_file:
            self.show_error("Спочатку відкрий файл, з якого ти тренував модель.")
            return

        if not self.trained_model:
            self.show_error("Модель ще не завантажена.")
            return

        if not hasattr(self.training_thread, "scaler_X") or not hasattr(self.training_thread, "scaler_y"):
            self.show_error("Скалери не завантажені. Файл моделі пошкоджений.")
            return

        try:
            # Читаємо дані
            df = pd.read_excel(self.selected_file) if self.selected_file.endswith((".xls", ".xlsx")) else pd.read_csv(
                self.selected_file)

            if self.target_column_name not in df.columns:
                self.show_error(f"У файлі немає цільової колонки '{self.target_column_name}'.")
                return

            # Готуємо X і y
            X = df.drop(columns=[self.target_column_name])
            y_real = df[self.target_column_name].values
            X_numeric = X.select_dtypes(include=[np.number])

            if X_numeric.empty:
                self.show_error("У файлі немає числових даних для передбачення.")
                return

            count = self.ui.model_5_prediction_count.value()
            X_numeric = X_numeric.head(count)
            y_real = y_real[:count]

            # Масштабування
            X_scaled = self.training_thread.scaler_X.transform(X_numeric)

            # Передбачення
            y_pred_scaled = self.trained_model.predict(X_scaled, verbose=0)
            y_pred = self.training_thread.scaler_y.inverse_transform(y_pred_scaled).flatten()

            text = f"Модель завантажена\nЦіль: {self.target_column_name}\n\n"
            text += f"Перші {len(y_pred)} передбачень:\n\n"
            for i, (real, pred) in enumerate(zip(y_real, y_pred), start=1):
                text += f"{i}: Real = {real:.4f}, Predicted = {pred:.4f}\n"

            self.ui.model_5_result.setText(text)

        except Exception as e:
            self.show_error(f"Помилка під час прогнозування:\n{e}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Завантажити модель", "", "H5 Files (*.h5)")
        if not file_path:
            return

        from tensorflow.keras.models import load_model
        try:
            model = load_model(file_path)
            self.trained_model = model
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити модель:\n{e}")
            return

        try:
            with h5py.File(file_path, 'r') as f:
                scaler_X = StandardScaler()
                scaler_X.mean_ = f['scaler_X_mean'][:]
                scaler_X.scale_ = f['scaler_X_scale'][:]
                scaler_X.var_ = scaler_X.scale_ ** 2
                scaler_X.n_features_in_ = len(scaler_X.mean_)

                scaler_y = StandardScaler()
                scaler_y.mean_ = f['scaler_y_mean'][:].reshape(-1, 1)
                scaler_y.scale_ = f['scaler_y_scale'][:].reshape(-1, 1)
                scaler_y.var_ = scaler_y.scale_ ** 2

                self.target_column_name = f.attrs.get('target_column', 'Unknown')

            if not self.training_thread:
                self.training_thread = GeneticTrainWorker("", "", 1, [], 0)
            self.training_thread.scaler_X = scaler_X
            self.training_thread.scaler_y = scaler_y

            self.ui.model_5_save.setEnabled(True)
            self.ui.model_5_container.setCurrentIndex(1)
            self.ui.model_5_result.setText(
                f"Модель завантажена:\n{os.path.basename(file_path)}\nЦіль: {self.target_column_name}\nГотова до прогнозів!"
            )
            self.predict_after_load()

        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося прочитати скалери:\n{e}")

    def reset_model(self):
        print("[INFO] Resetting model and UI")
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.requestInterruption()
            self.training_thread.wait()

        self.training_thread = None
        self.trained_model = None
        self.X_data = None
        self.selected_file = None
        self.target_column_name = None

        self.ui.model_5_result.clear()
        self.ui.model_5_file.setText("No file selected")
        self.ui.model_5_save.setEnabled(False)
        self.ui.model_5_start_learning.setEnabled(True)
        self.ui.model_5_start_learning.setText("Start learning")
        self.ui.model_5_container.setCurrentIndex(0)
        self.ui.model_5_progress.setValue(0)
        self.ui.model_5_progress.hide()
        self.ui.model_5_target_column.hide()
        self.ui.model_5_target_column.clear()

        for w in self.layer_widgets[:]:
            w.setParent(None)
            self.layer_widgets.remove(w)
        self.add_layer()

    def show_error(self, message):
        print(f"[ERROR] {message}")
        QMessageBox.critical(self, "Помилка", message)