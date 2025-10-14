from PyQt5.QtCore import QThread, pyqtSignal
from models.model_lab1 import regression_model


class RegressionWorker(QThread):
    finished = pyqtSignal(str, object)  # result_text + model
    progress = pyqtSignal(str)

    def __init__(self, filepath="../data/student_scores.xlsx", epochs=50, test_size=0.2, layers=None, result_size=4):
        super().__init__()
        self.filepath = filepath
        self.epochs = epochs
        self.test_size = test_size
        self.layers = layers
        self.result_size = result_size

    def run(self):
        try:
            self.progress.emit("=== Починаю навчання моделі ===")
            result_text, model = regression_model(
                filepath=self.filepath,
                epochs=self.epochs,
                test_size=self.test_size,
                layers=self.layers,
                result_size=self.result_size
            )
            self.progress.emit("=== Навчання завершене успішно ===")
            self.finished.emit(result_text, model)
        except Exception as e:
            self.progress.emit(f"Помилка: {e}")
