import time
import json
import os
import math
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
class ImageNet_CNN_Worker(QThread):
    """
    QThread-робітник для тренування моделі.
    Очікується структура:
    dataset_path/
       train/
         dogs/
         cats/
       val/
         dogs/
         cats/
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, object, dict)
    def __init__(self, dataset_path, epochs=10, img_size=128, batch_size=128, layers=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        self.layers = layers or []
        self.model = None
        self.class_indices = None
    # --------------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------------
    def build_model(self, input_shape, num_classes):
        """
        Простий, але ефективний CNN з BatchNorm і Dropout.
        Параметри dense-слоїв можна передати через self.layers (список dict з keys: units, activation).
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dropout(0.4))
        # custom dense layers
        for layer in self.layers:
            units = int(layer.get("units", 128))
            activation = layer.get("activation", "relu")
            model.add(Dense(units, activation=activation))
        # final classifier
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model
    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------
    def run(self):
        try:
            self.progress.emit("📥 Перевірка структури папок та підготовка даних...")
            # Очікуємо dataset_path має підпапки train/ і val/
            train_dir = os.path.join(self.dataset_path, "train")
            val_dir = os.path.join(self.dataset_path, "val")
            if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
                raise FileNotFoundError("Папки 'train' і 'val' мають існувати всередині вибраного каталогу.")
            # Аугментація для тренування
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode="nearest"
            )
            val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
            train = train_datagen.flow_from_directory(
                train_dir,
                target_size=(self.img_size, self.img_size),
                class_mode="categorical",
                batch_size=self.batch_size,
                shuffle=True
            )
            val = val_datagen.flow_from_directory(
                val_dir,
                target_size=(self.img_size, self.img_size),
                class_mode="categorical",
                batch_size=self.batch_size,
                shuffle=False
            )
            if train.samples == 0 or val.samples == 0:
                raise ValueError("Немає зображень у train або val. Перевірте структуру папок і файли.")
            self.class_indices = train.class_indices
            num_classes = len(self.class_indices)
            input_shape = (self.img_size, self.img_size, 3)
            self.progress.emit(f"📊 Класів: {num_classes} — {self.class_indices}")
            self.progress.emit(f"Train samples: {train.samples}, Val samples: {val.samples}")
            self.model = self.build_model(input_shape, num_classes)
            # підготовка callback'ів: збереження найкращої моделі по val_loss, зниження lr та рання зупинка
            tmp_model_path = os.path.join(self.dataset_path, "best_model_temp.h5")
            checkpoint = ModelCheckpoint(tmp_model_path, monitor="val_loss", save_best_only=True, verbose=0)
            reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0)
            early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)
            steps_per_epoch = math.ceil(train.samples / float(self.batch_size))
            validation_steps = math.ceil(val.samples / float(self.batch_size))
            start = time.time()
            # Тренуємо епохами, але емулюємо прогрес по-епохам (щоб повідомляти UI)
            history = self.model.fit(
                train,
                epochs=self.epochs,
                validation_data=val,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=[checkpoint, reduce_lr, early_stop],
                verbose=0
            )
            train_time = time.time() - start
            # Якщо callback зберіг кращу модель у тимчасовий файл, підвантажимо її
            if os.path.exists(tmp_model_path):
                try:
                    best = load_model(tmp_model_path)
                    self.model = best
                    # видаляти тимчасовий файл не обов'язково
                except Exception:
                    pass
            # Формування підсумкового повідомлення з останніх метрик
            last_acc = history.history.get("accuracy", [None])[-1]
            last_val_acc = history.history.get("val_accuracy", [None])[-1]
            last_loss = history.history.get("loss", [None])[-1]
            last_val_loss = history.history.get("val_loss", [None])[-1]
            done = (
                f"🎉 Навчання завершено! Час: {train_time:.2f} сек. "
                f"acc={last_acc:.4f} val_acc={last_val_acc:.4f} "
                f"loss={last_loss:.4f} val_loss={last_val_loss:.4f}"
            )
            # Повертаємо модель і class_indices
            self.finished.emit(done, self.model, self.class_indices)
        except Exception as e:
            # При помилці повертаємо текст помилки
            self.finished.emit(f"❌ Помилка: {e}", None, None)
    # --------------------------------------------------------------
    # SAVE MODEL + CLASSES
    # --------------------------------------------------------------
    @staticmethod
    def save_model_full(model, class_indices, path):
        """
        Зберегти модель (h5 або .keras) та файл classes.json поруч (path + "_classes.json").
        """
        # ensure dir exists
        dest_dir = os.path.dirname(path) or "."
        os.makedirs(dest_dir, exist_ok=True)
        model.save(path)
        with open(path + "_classes.json", "w", encoding="utf-8") as f:
            # зберігаємо mapping label->index
            json.dump(class_indices, f, ensure_ascii=False, indent=2)
    # --------------------------------------------------------------
    # LOAD MODEL + CLASSES
    # --------------------------------------------------------------
    @staticmethod
    def load_model_full(path):
        model = load_model(path)
        with open(path + "_classes.json", "r", encoding="utf-8") as f:
            loaded = json.load(f)
        # Приводимо значення до int
        class_indices = {k: int(v) for k, v in loaded.items()}
        return model, class_indices
    # --------------------------------------------------------------
    # PREDICT IMAGE
    # --------------------------------------------------------------
    @staticmethod
    def predict_image(model, class_indices, file_path, img_size=128):
        """
        Повертає (label, probability)
        class_indices: dict label -> index, як з flow_from_directory
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не знайдено: {file_path}")
        img = load_img(file_path, target_size=(img_size, img_size))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = model.predict(arr)[0]
        idx = int(np.argmax(pred))
        # інвертуємо class_indices (index -> label)
        classes = {v: k for k, v in class_indices.items()}
        label = classes.get(idx, "unknown")
        prob = float(pred[idx])
        return label, prob