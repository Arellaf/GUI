from PyQt5.QtCore import QThread, pyqtSignal
from models.model_lab2 import classification_model

class ClassificationWorker(QThread):
    finished = pyqtSignal(str, object)
    progress = pyqtSignal(str)

    def __init__(self, filepath, target_column, epochs=50, test_size=0.2, layers=None, result_size=4):
        super().__init__()
        self.filepath = filepath
        self.target_column = target_column
        self.epochs = epochs
        self.test_size = test_size
        self.layers = layers
        self.result_size = result_size

    def run(self):
        try:
            self.progress.emit("=== Починаю навчання моделі класифікації ===")
            result_text, model = classification_model(
                filepath=self.filepath,
                target_column=self.target_column,
                epochs=self.epochs,
                test_size=self.test_size,
                layers=self.layers,
                result_size=self.result_size
            )
            self.progress.emit("=== Навчання завершене успішно ===")
            self.finished.emit(result_text, model)
        except Exception as e:
            self.progress.emit(f"Помилка: {e}")
