from workers.image_cnn_worker import ImageNet_CNN_Worker
import os
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QSpinBox
class Model9Page(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.dataset_path = None
        self.trained_model = None
        self.class_indices = None
        self.worker = None
        self.custom_layers = []
        # кнопки (припускаю, що ui вже містить ці елементи)
        self.ui.upload_my_data_9.clicked.connect(self.load_dataset)
        self.ui.learn_btn_9.clicked.connect(self.run_training)
        self.ui.safe_model_btn_9.clicked.connect(self.save_model)
        self.ui.upload_my_model_9.clicked.connect(self.load_model)
        self.ui.add_layer_btn_9.clicked.connect(self.add_layer)
        # self.ui.predict_btn_9.clicked.connect(self.select_and_predict_image)
        self.add_layer() # default layer
    # ---------------------------------------------------------
    def load_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Оберіть папку датасету")
        if folder:
            # перевіряємо наявність train/ val/ всередині
            train_dir = os.path.join(folder, "train")
            val_dir = os.path.join(folder, "val")
            if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
                QMessageBox.warning(self, "Помилка", "Папка датасету має містити підпапки 'train' та 'val'.")
                return
            self.dataset_path = folder
            self.ui.model_9_file.setText(folder)
            QMessageBox.information(self, "OK", f"Вибрано датасет:\n{folder}")
    # ---------------------------------------------------------
    def add_layer(self):
        row = QWidget()
        layout = QHBoxLayout(row)
        spin = QSpinBox()
        spin.setRange(8, 2048)
        spin.setValue(128)
        combo = QComboBox()
        combo.addItems(["relu", "sigmoid", "tanh"])
        btn = QPushButton("X")
        layout.addWidget(spin)
        layout.addWidget(combo)
        layout.addWidget(btn)
        container = self.ui.add_layers_model_9.layout() or QVBoxLayout(self.ui.add_layers_model_9)
        self.ui.add_layers_model_9.setLayout(container)
        container.addWidget(row)
        cfg = {"units": 128, "activation": "relu"}
        self.custom_layers.append(cfg)
        spin.valueChanged.connect(lambda v, c=cfg: c.update({"units": v}))
        combo.currentTextChanged.connect(lambda v, c=cfg: c.update({"activation": v}))
        btn.clicked.connect(lambda: self.remove_layer(row, cfg))
    # ---------------------------------------------------------
    def remove_layer(self, widget, cfg):
        try:
            self.custom_layers.remove(cfg)
        except ValueError:
            pass
        widget.setParent(None)
    # ---------------------------------------------------------
    def run_training(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Помилка", "Оберіть датасет!")
            return
        try:
            epochs = int(self.ui.epochs_size_9.text() or 5)
        except Exception:
            epochs = 5
        self.worker = ImageNet_CNN_Worker(
            self.dataset_path,
            epochs=epochs,
            img_size=128,
            batch_size=32,
            layers=self.custom_layers
        )
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.training_finished)
        self.worker.start()
    def update_status(self, msg):
        self.ui.result_model_9.setText(msg)
    def training_finished(self, text, model, classes):
        self.trained_model = model
        self.class_indices = classes
        self.ui.result_model_9.setText(text)
    # ---------------------------------------------------------
    def save_model(self):
        if self.trained_model is None:
            QMessageBox.warning(self, "Помилка", "Модель ще не навчена!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти модель", "", "Keras (*.h5 *.keras)")
        if path:
            try:
                ImageNet_CNN_Worker.save_model_full(self.trained_model, self.class_indices, path)
                QMessageBox.information(self, "OK", "Модель збережена!")
            except Exception as e:
                QMessageBox.critical(self, "Помилка при збереженні", str(e))
    # ---------------------------------------------------------
    def load_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "Завантажити модель", "", "Keras (*.h5 *.keras)")
        if file:
            try:
                self.trained_model, self.class_indices = ImageNet_CNN_Worker.load_model_full(file)
                QMessageBox.information(self, "OK", "Модель завантажена!")
            except Exception as e:
                QMessageBox.critical(self, "Помилка", str(e))
    # ---------------------------------------------------------
    def predict_test_image(self, image_path):
        """
        Додатковий зручний метод — передбачити локальне фото (test_image.jpg).
        Повертає (label, probability) або викликає QMessageBox при помилці.
        """
        if self.trained_model is None or self.class_indices is None:
            QMessageBox.warning(self, "Помилка", "Модель не завантажена і не навчена.")
            return None
        try:
            label, prob = ImageNet_CNN_Worker.predict_image(self.trained_model, self.class_indices, image_path, img_size=128)
            QMessageBox.information(self, "Результат", f"Файл: {os.path.basename(image_path)}\nКлас: {label}\nЙмовірність: {prob:.4f}")
            return label, prob
        except Exception as e:
            QMessageBox.critical(self, "Помилка прогнозу", str(e))
            return None
    def select_and_predict_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Оберіть фото", "", "Images (*.jpg *.png *.jpeg)")
        if file:
            self.predict_test_image(file)