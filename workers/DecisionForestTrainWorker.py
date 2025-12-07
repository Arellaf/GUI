from workers.BaseTrainWorker import BaseTrainWorker
from PyQt5.QtCore import pyqtSignal
import ydf                                      # <-- НОВА БІБЛІОТЕКА
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


class DecisionForestTrainWorker(BaseTrainWorker):
    finished = pyqtSignal(object, float, object, object, object, object, object, float, object, object, object)

    def __init__(self, file_path, target_column, model_type, max_depth, num_trees, test_size=0.2, task="REGRESSION"):
        super().__init__(file_path, target_column)
        self.model_type = model_type          # "RANDOM_FOREST", "GRADIENT_BOOSTED_TREES", "CART"
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.test_size = test_size
        self.task = task.upper()              # "REGRESSION" / "CLASSIFICATION"

    def run(self):
        try:
            #Завантаження даних
            if self.file_path.endswith((".xls", ".xlsx")):
                data = pd.read_excel(self.file_path)
            elif self.file_path.endswith(".csv"):
                data = pd.read_csv(self.file_path)
            else:
                raise ValueError("Only .xls, .xlsx, and .csv formats are supported")

            data = data.dropna(axis=1, how="all").dropna(axis=0, how="all")
            if self.target_column not in data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

            mappings = {}

            stratify = data[self.target_column] if self.task == "CLASSIFICATION" else None
            train_df, test_df = train_test_split(
                data, test_size=self.test_size, random_state=42, stratify=stratify
            )

            task_ydf = ydf.Task.REGRESSION if self.task == "REGRESSION" else ydf.Task.CLASSIFICATION

            if self.model_type.upper() == "RANDOM_FOREST":
                learner = ydf.RandomForestLearner(
                    label=self.target_column,
                    task=task_ydf,
                    max_depth=self.max_depth,
                    num_trees=self.num_trees,
                )
            elif self.model_type.upper() == "GRADIENT_BOOSTED_TREES":
                learner = ydf.GradientBoostedTreesLearner(
                    label=self.target_column,
                    task=task_ydf,
                    max_depth=self.max_depth,
                    num_trees=self.num_trees,
                    shrinkage=0.1,
                )
            elif self.model_type.upper() == "CART":
                learner = ydf.CartLearner(
                    label=self.target_column,
                    task=task_ydf,
                    max_depth=self.max_depth,
                )
            else:
                raise ValueError("model_type must be RANDOM_FOREST, GRADIENT_BOOSTED_TREES or CART")

            model = learner.train(train_df, verbose=False)

            y_pred = model.predict(test_df)
            y_test = test_df[self.target_column].values

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred) if self.task == "REGRESSION" else None

            history = None
            x_test = test_df.drop(columns=self.target_column).values
            X = data.drop(columns=self.target_column).values

            if not self.isInterruptionRequested():
                self.finished.emit(
                    model,
                    mae,
                    history,
                    x_test,
                    y_test,
                    y_pred,
                    X,
                    r2 or 0.0,
                    None,
                    None,
                    mappings
                )

        except Exception as e:
            self.message.emit(f"YDF Decision Forest training error: {str(e)}")
            self.finished.emit(None, 0.0, None, None, None, None, None, 0.0, None, None, None)