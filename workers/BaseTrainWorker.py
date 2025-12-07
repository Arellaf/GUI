from PyQt5.QtCore import QThread, pyqtSignal

class BaseTrainWorker(QThread):
    """Abstract base class for training workers."""

    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished = pyqtSignal(object, float, object, object, object, object)
    # weights, mse, ga_instance, scaler_X, scaler_y, X_scaled

    def __init__(self, file_path, target_column, epochs=30, layers=None, dropout=0.2, rules=8):
        super().__init__()
        self.file_path = file_path
        self.target_column = target_column
        self.epochs = epochs
        self.layers = layers
        self.dropout = dropout
        self.rules = rules

    def stop_requested(self):
        return self.isInterruptionRequested()

    def run(self):
        raise NotImplementedError("Subclasses must implement this method.")