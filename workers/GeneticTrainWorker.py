import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pygad.kerasga
from workers.BaseTrainWorker import BaseTrainWorker

class GeneticTrainWorker(BaseTrainWorker):

    def __init__(
        self,
        file_path,
        target_column,
        population_size=100,
        generations=50,
        mutation_percent=2.0,
        layers=None,
        adam_epochs=30,
        noise_scale=0.02,
        patience=8,
        mutation_increase=1.5,
        seed=None,
    ):
        super().__init__(file_path, target_column)
        self.population_size = population_size
        self.generations = generations
        self.mutation_percent = float(mutation_percent)
        self.layers = layers or [(32, 'relu'), (16, 'relu')]
        self.adam_epochs = adam_epochs
        self.noise_scale = noise_scale
        self.patience = patience
        self.mutation_increase = mutation_increase
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
            tf.config.experimental.enable_op_determinism()

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.y_train_original = None
        self.model = None
        self.best_history = []

    def build_model(self, input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        for neurons, activation in self.layers:
            model.add(tf.keras.layers.Dense(neurons, activation=activation))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def run(self):
        try:
            print("[INFO] Завантаження та обробка даних...")
            df = pd.read_csv(self.file_path) if self.file_path.endswith('.csv') else pd.read_excel(self.file_path)

            df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
            if self.target_column not in df.columns:
                self.message.emit(f"Колонка '{self.target_column}' не знайдена!")
                return

            df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')

            for col in df.select_dtypes(include=['object']).columns:
                if col == self.target_column:
                    continue
                df[col] = df[col].astype(str).str.strip().str.lower()
                mapping = {v: i + 1 for i, v in enumerate(sorted(df[col].unique()))}
                df[col] = df[col].map(mapping)

            if df.isnull().any().any():
                self.message.emit("Знайдено пропуски після обробки даних!")
                return

            X = df.drop(columns=[self.target_column]).astype(np.float32)
            y = df[self.target_column].values.astype(np.float32)

            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

            self.X_train = X_scaled
            self.y_train = y_scaled
            self.y_train_original = y.copy()

            self.model = self.build_model((X_scaled.shape[1],))

            # Adam pre-training
            print(f"[INFO] Попереднє навчання Adam ({self.adam_epochs} епох)...")
            self.model.fit(self.X_train, self.y_train, epochs=self.adam_epochs, verbose=1, batch_size=64)

            pred_adam = self.model.predict(self.X_train, verbose=0)
            pred_adam_orig = self.scaler_y.inverse_transform(pred_adam).flatten()
            mse_adam = np.mean((self.y_train_original - pred_adam_orig) ** 2)
            print(f"[INFO] MSE після Adam: {mse_adam:.6f}")

            # Підготовка популяції
            base_weights = pygad.kerasga.model_weights_as_vector(self.model)
            num_genes = len(base_weights)

            population = np.tile(base_weights, (self.population_size, 1)).astype(np.float32)
            population += np.random.normal(0, self.noise_scale, population.shape)
            population[0] = base_weights  # еліта

            best_mse_global = mse_adam
            self.best_history = [best_mse_global]

            print(f"[INFO] Запуск швидкого ГА: {self.generations} поколінь, популяція = {self.population_size}")

            for gen in range(1, self.generations + 1):
                # Батч-предикт для всієї популяції
                all_preds = np.zeros((self.population_size, len(self.X_train)), dtype=np.float32)

                for i, weights_vec in enumerate(population):
                    weights_matrix = pygad.kerasga.model_weights_as_matrix(self.model, weights_vec)
                    self.model.set_weights(weights_matrix)
                    all_preds[i] = self.model.predict(self.X_train, verbose=0, batch_size=4096).flatten()

                # MSE
                preds_orig = self.scaler_y.inverse_transform(all_preds.reshape(-1, 1)).reshape(self.population_size, -1)
                errors = self.y_train_original - preds_orig
                mses = np.mean(errors ** 2, axis=1)
                best_idx = np.argmin(mses)
                best_mse_gen = mses[best_idx]

                if best_mse_gen < best_mse_global:
                    best_mse_global = best_mse_gen

                print(f"Покоління {gen:2d}: best MSE = {best_mse_gen:.6f} | avg = {np.mean(mses):.6f}")
                self.best_history.append(best_mse_gen)

                # Прогрес
                try:
                    self.progress.emit(int(gen / self.generations * 100))
                except:
                    pass

                # Адаптивна мутація
                if len(self.best_history) > self.patience + 1:
                    recent = self.best_history[-(self.patience + 1):]
                    if min(recent[:-1]) <= recent[-1]:
                        new_mut = min(50.0, self.mutation_percent * self.mutation_increase)
                        if abs(new_mut - self.mutation_percent) > 1e-6:
                            print(f"[INFO] Мутація ↑ {self.mutation_percent:.1f}% → {new_mut:.1f}%")
                            self.mutation_percent = new_mut

                # Елітарність
                elite_count = max(1, int(self.population_size * 0.1))
                elite_idx = np.argsort(mses)[:elite_count]

                # Батьки — топ-кращі
                num_parents = max(8, self.population_size // 4)
                parent_indices = np.argsort(mses)[:num_parents]
                parents = population[parent_indices]

                # Якщо батьків менше 2 — дублюємо найкращого
                if len(parents) < 2:
                    parents = np.tile(population[best_idx], (8, 1))

                # Кросовер + мутація
                offspring = []
                target_offspring = self.population_size - elite_count

                while len(offspring) < target_offspring:
                    p1_idx, p2_idx = np.random.choice(len(parents), size=2, replace=False)
                    p1, p2 = parents[p1_idx], parents[p2_idx]

                    mask = np.random.random(num_genes) < 0.5
                    child = np.where(mask, p1, p2)

                    mut_mask = np.random.random(num_genes) < (self.mutation_percent / 100.0)
                    child[mut_mask] += np.random.normal(0, 0.05, np.sum(mut_mask))

                    offspring.append(child)

                # Нова популяція
                new_population = np.vstack([
                    population[elite_idx],
                    np.array(offspring[:target_offspring], dtype=np.float32)
                ])
                population = new_population

            # Фінальна найкраща модель
            final_weights = pygad.kerasga.model_weights_as_matrix(self.model, population[best_idx])
            self.model.set_weights(final_weights)

            final_pred = self.model.predict(self.X_train, verbose=0)
            final_pred_orig = self.scaler_y.inverse_transform(final_pred).flatten()
            self.last_mse = np.mean((self.y_train_original - final_pred_orig) ** 2)

            print(f"\n[ГОТОВО] Фінальне MSE: {self.last_mse:.6f}")
            print(f"[ГОТОВО] Покращення: {mse_adam:.6f} → {self.last_mse:.6f}")

            weights_list = [w.tolist() for w in final_weights]
            self.finished.emit(
                weights_list,
                self.last_mse,
                None,
                self.scaler_X,
                self.scaler_y,
                self.X_train
            )

        except Exception as e:
            import traceback
            err = f"Помилка: {e}\n{traceback.format_exc()}"
            print(err)
            self.message.emit(err)
            self.progress.emit(0)