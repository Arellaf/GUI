from PyQt5.QtCore import QThread, pyqtSignal
from models.model_lab3 import neurofuzzy_model

class NeuroFuzzyWorker(QThread):
    finished = pyqtSignal(str, object, object, object, object)
    progress = pyqtSignal(str)

    def __init__(self, filepath, target_column,
                 epochs=50, test_size=0.2, fuzzy_layers=None,
                 result_size=4):
        super().__init__()
        self.filepath = filepath
        self.target_column = target_column
        self.epochs = epochs
        self.test_size = test_size
        self.fuzzy_layers = fuzzy_layers
        self.result_size = result_size

    def run(self):
        try:
            self.progress.emit("=== Начинаю обучение Neuro-Fuzzy модели ===")
            result_text, model, y_real, y_pred, plot_path = neurofuzzy_model(
                filepath=self.filepath,
                target_column=self.target_column,
                epochs=self.epochs,
                test_size=self.test_size,
                fuzzy_layers=self.fuzzy_layers,
                result_size=self.result_size
            )
            self.progress.emit("=== Обучение завершено успешно ===")
            self.finished.emit(result_text, model, y_real, y_pred, plot_path)
        except Exception as e:
            self.progress.emit(f"Ошибка: {e}")