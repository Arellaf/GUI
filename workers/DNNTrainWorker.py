from models.model_lab6 import train_dnn_model
from PyQt5.QtCore import QThread, pyqtSignal

class BaseTrainWorker(QThread):
    """Abstract base class for training workers."""

    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished = pyqtSignal(object, float, object, object, object, object)

    def __init__(self, file_path, test_size, target_column, epochs=30, layers=None, dropout=0.2, rules=8):
        super().__init__()
        self.test_size = test_size
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

class DNNTrainWorker(BaseTrainWorker):
    finished = pyqtSignal(object, float, object, object, object, object, object, float, object, object, object)

    def init(self, file_path, target_column, epochs, layers, dropout, test_size=0.2):
        super().init(file_path, target_column, epochs, layers, dropout)
        self.test_size = test_size
        self.scaler_X = None
        self.scaler_y = None

    def run(self):
        try:
            model, mae, history, x_test, y_test, y_pred, X, r2, scaler_X, scaler_y = train_dnn_model(
                file=self.file_path,
                target_column=self.target_column,
                epochs=self.epochs,
                layers=self.layers,
                dropout=self.dropout,
                test_size=self.test_size,
                progress_callback=self.progress,
                stop_callback=self.stop_requested,
                return_scalers=True
            )

            if not self.isInterruptionRequested():
                self.finished.emit(
                    model, mae, history, x_test, y_test, y_pred, X, r2, scaler_X, scaler_y, None
                )

        except Exception as e:
            self.message.emit(f"DNN training error: {str(e)}")
            self.finished.emit(None, 0.0, None, None, None, None, None, 0.0, None, None, None)