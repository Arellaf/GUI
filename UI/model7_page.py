import json
import os
import logging
import numpy as np
import pandas as pd
import ydf
from PyQt5.QtWidgets import QWidget, QFileDialog, QVBoxLayout, QMessageBox
from sklearn.metrics import r2_score, mean_absolute_error
import h5py
import zipfile  # для пакування YDF у ZIP
from workers.DecisionForestTrainWorker import DecisionForestTrainWorker

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("Model7Page")


class Model7Page(QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.selected_file = None
        self.training_thread = None
        self.trained_model = None
        self.X_data = None
        self.target_column_name = None
        self.mappings = {}
        self.params = None

        log.info("Ініціалізація Decision Forest (YDF) — H5 формат")

        # Заповнення типів моделей
        self.ui.model_7_type.clear()
        self.model_types = {
            "Випадковий ліс": "RANDOM_FOREST",
            "Градієнтний бустинг": "GRADIENT_BOOSTED_TREES",
            "Дерево рішень (CART)": "CART"
        }
        for display_name in self.model_types:
            self.ui.model_7_type.addItem(display_name)
        self.ui.model_7_type.setCurrentIndex(0)

        # Підключення кнопок
        self.ui.model_7_start_learning.clicked.connect(self.start_learning)
        self.ui.model_7_upload_data.clicked.connect(self.select_file)
        self.ui.model_7_save.clicked.connect(self.save_model)
        self.ui.model_7_upload_model.clicked.connect(self.load_model)
        self.ui.model_7_reset.clicked.connect(self.reset_model)

        self.ui.model_7_save.setEnabled(False)
        self.ui.model_7_progress.setValue(0)
        self.ui.model_7_progress.setFormat("Навчання: %p%")
        self.ui.model_7_progress.hide()
        self.ui.model_7_target_column.hide()

        self.ui.stackedWidget_btn_next_11.clicked.connect(lambda: self.ui.model_7_container.setCurrentIndex(1))
        self.ui.stackedWidget_btn_back_7.clicked.connect(lambda: self.ui.model_7_container.setCurrentIndex(0))
        self.ui.stackedWidget_btn_next_12.clicked.connect(lambda: self.ui.model_7_container.setCurrentIndex(2))

        # Приховуємо графік, якщо він є
        if hasattr(self.ui, 'model_7_graphic'):
            self.ui.model_7_graphic.hide()

        # Також можна прибрати відповідний layout, якщо він є
        if hasattr(self.ui, 'model_7_graphic_layout'):
            self.ui.model_7_graphic_layout.setEnabled(False)
            self.ui.model_7_graphic_layout.setVisible(False)

        log.info("Model7Page готова — H5 режим (без графіка)")

    def select_file(self):
        log.info("Відкриття діалогу вибору файлу")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть файл з даними", "",
            "Excel/CSV (*.xls *.xlsx *.csv);;All Files (*)"
        )
        if not file_path:
            log.info("Вибір скасовано")
            return

        self.selected_file = file_path
        self.ui.model_7_file.setText(f"Файл: {os.path.basename(file_path)}")
        log.info(f"Вибрано: {file_path}")

        try:
            df_preview = pd.read_excel(file_path, nrows=5) if file_path.endswith((".xls", ".xlsx")) else pd.read_csv(
                file_path, nrows=5)
            numeric_cols = df_preview.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                self.show_error("Немає числових колонок")
                return

            self.ui.model_7_target_column.clear()
            self.ui.model_7_target_column.addItems(numeric_cols)
            self.ui.model_7_target_column.setEnabled(True)
            self.ui.model_7_target_column.show()
            log.info(f"Доступно цільових колонок: {len(numeric_cols)}")
        except Exception as e:
            log.error(f"Помилка читання: {e}")
            self.show_error(f"Не вдалося прочитати файл: {e}")

    def start_learning(self):
        if not self.selected_file:
            self.show_error("Оберіть файл")
            return
        if not self.ui.model_7_target_column.currentText():
            self.show_error("Оберіть цільову колонку")
            return

        try:
            max_depth = int(self.ui.model_7_tree_deep_input.text() or "0") or None
            num_trees = int(self.ui.model_7_tree_count_input.text() or "100")
            model_type = self.model_types[self.ui.model_7_type.currentText()]
            test_size = float(self.ui.model_7_test_size.text() or "0.2")
        except ValueError:
            self.show_error("Неправильні числа")
            return

        self.target_column_name = self.ui.model_7_target_column.currentText()
        log.info(f"Запуск навчання YDF: {model_type}, дерев: {num_trees}, глибина: {max_depth}")

        self.ui.model_7_progress.setValue(0)
        self.ui.model_7_progress.show()
        self.ui.model_7_start_learning.setEnabled(False)
        self.ui.model_7_start_learning.setText("Навчання...")

        self.training_thread = DecisionForestTrainWorker(
            file_path=self.selected_file,
            target_column=self.target_column_name,
            model_type=model_type,
            max_depth=max_depth,
            num_trees=num_trees,
            test_size=test_size
        )
        self.training_thread.message.connect(self.show_error)
        self.training_thread.finished.connect(self.training_done)
        self.training_thread.start()

    def training_done(self, model, mae, history, x_test, y_test, y_pred, X, r2, _, __, mappings):
        self.ui.model_7_progress.hide()
        self.ui.model_7_start_learning.setEnabled(True)
        self.ui.model_7_start_learning.setText("Запустити навчання")

        if model is None:
            log.error("Навчання провалилося")
            return

        log.info(f"Навчання завершено: MAE={mae:.4f}, R²={r2:.4f}")
        self.trained_model = model
        self.X_data = pd.DataFrame(X)
        self.mappings = mappings or {}
        self.params = {
            "model_type": self.training_thread.model_type,
            "max_depth": self.training_thread.max_depth,
            "num_trees": self.training_thread.num_trees,
            "mae": mae,
            "r2": r2
        }

        self.ui.model_7_save.setEnabled(True)
        self.ui.model_7_container.setCurrentIndex(1)
        self.show_predictions(mae, None, y_test, y_pred, r2)

    def show_predictions(self, mae, history, y_test, y_pred, r2):
        if y_test is None:
            return
        n = min(self.ui.model_7_prediction_count.value(), len(y_pred))
        sample = "\n".join([f"{i + 1:2d}: Реал: {y_test[i]:8.4f} → {y_pred[i]:8.4f}"
                            for i in range(n)])

        model_name = next(k for k, v in self.model_types.items() if v == self.params["model_type"])
        info = f"""YDF Decision Forest — Навчання завершено!

Ціль: {self.target_column_name}
Модель: {model_name}
Дерев: {self.params['num_trees']:,}
Глибина: {self.params['max_depth'] or 'без обмежень'}
MAE: {mae:.4f}
R²: {r2:.4f}

Перші {n} прогнозів:
{sample}"""

        self.ui.model_7_result.setText(info)

    def save_model(self):
        if not self.trained_model:
            return QMessageBox.critical(self, "Помилка", "Немає навченою моделі")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти модель YDF", "", "H5 Files (*.h5)"
        )
        if not file_path:
            return
        if not file_path.endswith(".h5"):
            file_path += ".h5"

        try:
            import tempfile
            import zipfile

            with tempfile.TemporaryDirectory() as tmpdir:
                # 1. Зберігаємо YDF-модель у нативному форматі
                self.trained_model.save(tmpdir)
                log.info(f"YDF модель збережена у тимчасову папку: {tmpdir}")

                # 2. Пакуемо всю папку у ZIP
                zip_path = os.path.join(tmpdir, "model.ydf.zip")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
                    for root, _, files in os.walk(tmpdir):
                        for file in files:
                            full_path = os.path.join(root, file)
                            arcname = os.path.relpath(full_path, tmpdir)
                            z.write(full_path, arcname)

                # 3. Створюємо .h5 і записуємо ZIP + метадані
                with open(zip_path, "rb") as zf:
                    zip_data = zf.read()

                with h5py.File(file_path, "w") as f:
                    # Записуємо ZIP як бінарні дані
                    f.create_dataset("ydf_model_data", data=np.void(zip_data))

                    # Записуємо метадані ТІЛЬКИ як UTF-8 байти — без NULLs!
                    def safe_str(s):
                        return str(s).encode('utf-8')

                    f.attrs["framework"] = safe_str("YDF")
                    f.attrs["target_column"] = safe_str(self.target_column_name or "Unknown")
                    f.attrs["model_type"] = safe_str(self.params.get("model_type", "UNKNOWN"))
                    f.attrs["num_trees"] = int(self.params.get("num_trees", 100))
                    f.attrs["max_depth"] = int(self.params["max_depth"]) if self.params["max_depth"] else 0
                    f.attrs["mae"] = float(self.params.get("mae", 0.0))
                    f.attrs["r2"] = float(self.params.get("r2", 0.0))

            log.info(f"Модель успішно збережена у .h5: {file_path}")
            QMessageBox.information(self, "Готово", f"Модель збережена!\n{os.path.basename(file_path)}")

        except Exception as e:
            log.error(f"Помилка збереження: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти:\n{e}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити модель", "", "H5 Files (*.h5)"
        )
        if not file_path:
            return

        try:
            import tempfile
            import zipfile

            with h5py.File(file_path, "r") as f:
                # ВИПРАВЛЕНА ПЕРЕВІРКА: порівнюємо з bytes або декодуємо
                if "framework" not in f.attrs:
                    raise ValueError("Це не YDF-модель у форматі .h5")

                framework = f.attrs["framework"]
                # Декодуємо, якщо це bytes, інакше залишаємо як є
                if isinstance(framework, bytes):
                    framework = framework.decode('utf-8')

                if framework != "YDF":
                    raise ValueError("Це не YDF-модель у форматі .h5")

                # Витягуємо ZIP
                zip_data = bytes(f["ydf_model_data"][()])

                # Розпаковуємо у тимчасову папку
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "model.zip")
                    with open(zip_path, "wb") as zf:
                        zf.write(zip_data)

                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(tmpdir)

                    # Шукаємо папку з моделлю
                    model_dir = tmpdir
                    for root, dirs, files in os.walk(tmpdir):
                        if any(f.endswith(".done") or f.endswith(".ydf") for f in files):
                            model_dir = root
                            break

                    # Завантажуємо YDF-модель
                    self.trained_model = ydf.load_model(model_dir)
                    log.info("YDF модель успішно завантажена з .h5")

                # Зчитуємо метадані
                def safe_decode(b):
                    if isinstance(b, bytes):
                        return b.decode('utf-8')
                    elif isinstance(b, (int, float)):
                        return b
                    else:
                        return str(b)

                self.target_column_name = safe_decode(f.attrs.get("target_column", "Unknown"))
                model_type = safe_decode(f.attrs.get("model_type", "UNKNOWN"))

                self.params = {
                    "model_type": model_type,
                    "num_trees": int(f.attrs.get("num_trees", 100)),
                    "max_depth": int(f.attrs.get("max_depth", 0)) if f.attrs.get("max_depth", 0) != 0 else None,
                    "mae": float(f.attrs.get("mae", 0.0)),
                    "r2": float(f.attrs.get("r2", 0.0))
                }

            self.ui.model_7_save.setEnabled(True)
            self.ui.model_7_container.setCurrentIndex(1)
            self.ui.model_7_result.setText(f"Модель YDF завантажена з:\n{os.path.basename(file_path)}")
            QMessageBox.information(self, "Готово", "Модель успішно завантажена!")

            if self.selected_file:
                self._predict_with_loaded_model()

        except Exception as e:
            log.error(f"Помилка завантаження .h5: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити:\n{e}")

    def _predict_with_loaded_model(self):
        if not self.selected_file or not self.trained_model:
            return

        try:
            df = pd.read_excel(self.selected_file) if self.selected_file.endswith((".xls", ".xlsx")) else pd.read_csv(
                self.selected_file)
            df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
            X_raw = df.drop(columns=[self.target_column_name], errors="ignore")

            # YDF сам усе обробляє
            y_pred = self.trained_model.predict(X_raw)
            y_real = df[self.target_column_name].values if self.target_column_name in df.columns else None

            n = min(self.ui.model_7_prediction_count.value(), len(y_pred))
            sample = "\n".join([f"{i + 1}: → {y_pred[i]:.4f}" for i in range(n)])
            if y_real is not None:
                mae = mean_absolute_error(y_real[:n], y_pred[:n])
                r2 = r2_score(y_real, y_pred)
                sample = "\n".join([f"{i + 1}: {y_real[i]:.4f} → {y_pred[i]:.4f}" for i in range(n)])
                text = f"Прогноз завершено\nMAE: {mae:.4f} | R²: {r2:.4f}\n\n{sample}"
            else:
                text = f"Прогноз (без реальних значень):\n{sample}"

            self.ui.model_7_result.setText(text)

        except Exception as e:
            self.ui.model_7_result.append(f"\nПомилка прогнозу: {e}")

    def reset_model(self):
        log.info("Скидання Model7")
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.requestInterruption()

        self.training_thread = None
        self.trained_model = None
        self.X_data = None
        self.target_column_name = None
        self.mappings = {}
        self.params = None

        self.ui.model_7_result.clear()
        self.ui.model_7_file.setText("")
        self.ui.model_7_save.setEnabled(False)
        self.ui.model_7_start_learning.setEnabled(True)
        self.ui.model_7_start_learning.setText("Запустити навчання")
        self.ui.model_7_container.setCurrentIndex(0)
        self.ui.model_7_progress.hide()
        self.ui.model_7_target_column.hide()
        self.ui.model_7_target_column.clear()

    def show_error(self, message):
        log.error(f"ПОМИЛКА: {message}")
        self.reset_model()
        QMessageBox.critical(self, "Помилка", message)