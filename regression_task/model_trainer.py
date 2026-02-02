from utils import setup_logger
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from typing import Any
import os
import pickle
import json


class ModelTrainer:
    """Класс для обучения, оценки и сохранения модели Ridge."""

    def __init__(self, data_dir: str, resources_dir: str = "resources") -> None:
        """Инициализация ModelTrainer.

        Аргументы:
            data_dir: Путь к директории с файлами `X_data.npy` и `y_data.npy`.
            resources_dir: Директория для сохранения модели и метаданных.
        """
        self.data_dir = data_dir
        self.resources_dir = resources_dir
        self.logger = setup_logger(self.__class__.__name__)

        self.model: Ridge | None = None
        self.metadata: dict[str, Any] | None = None

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> None:
        """Загружает данные из файлов `X_data.npy` и `y_data.npy`.

        Аргументы:
            tuple[np.ndarray, np.ndarray]: Загруженные X и y.

        Исключения:
            FileNotFoundError: Если файлы данных не найдены.
        """
        X_path = os.path.join(self.data_dir, "X_data.npy")
        y_path = os.path.join(self.data_dir, "y_data.npy")

        if not os.path.exists(X_path) or not os.path.exists(y_path):
            self.logger.error("Файлы данных не найдены в %s", self.data_dir)
            raise FileNotFoundError(
                f"Не найдены файлы данных в директории {self.data_dir}"
            )

        self.X = np.load(X_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)

        self.logger.info(
            "Данные загружены | X.shape = %s | y.shape = %s",
            self.X.shape,
            self.y.shape,
        )

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """Делит данные на обучающую и тестовую выборки.

        Аргументы:
            test_size: Доля тестовой выборки.
            random_state: Seed для воспроизводимости.

        Исключения:
            ValueError: Если данные не загружены.
        """
        if self.X is None or self.y is None:
            self.logger.error("Попытка split_data без загруженных данных")
            raise ValueError("Сначала вызовите load_data()")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
        )

        self.logger.info(
            "Данные разделены | train = %d (%.1f%%) | test = %d (%.1f%%)",
            len(self.X_train),
            len(self.X_train) / len(self.X) * 100,
            len(self.X_test),
            len(self.X_test) / len(self.X) * 100,
        )

    def train_model(
        self,
        param_grid: dict[str, list[Any]] | None = None,
        cv_folds: int = 5,
    ) -> tuple[Ridge, dict[str, Any]]:
        """Обучает Ridge.

        Аргументы:
            param_grid: Гиперпараметры для модели.
            cv_folds: Количество фолдов для кросс-валидации.

        Вернёт:
            Обученная модель и метаданные.

        Исключения:
            ValueError: Если данные не разделены.
        """
        if self.X_train is None or self.y_train is None:
            self.logger.error("Попытка обучения без split_data")
            raise ValueError("Сначала вызовите split_data()")

        self.logger.info("Начато обучение модели")

        if param_grid is not None and len(param_grid) > 0:
            self.logger.info(
                "GridSearchCV | folds = %d | param_grid = %s",
                cv_folds,
                param_grid,
            )

            base_model = Ridge(random_state=42)
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring="r2",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(self.X_train, self.y_train)

            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            self.logger.info(
                "Лучшие параметры: %s | CV R2 = %.4f",
                best_params,
                best_score,
            )
        else:
            best_params = {}
            self.model = Ridge(random_state=42)
            self.model.fit(self.X_train, self.y_train)
            self.logger.info("Модель обучена без подбора гиперпараметров")

        train_metrics = self.evaluate_model(self.X_train, self.y_train)
        test_metrics = self.evaluate_model(self.X_test, self.y_test)

        self.logger.info(
            "Train metrics | MSE = %.4f | MAE = %.4f | R2 = %.4f",
            train_metrics["mse"],
            train_metrics["mae"],
            train_metrics["r2"],
        )
        self.logger.info(
            "Test metrics  | MSE = %.4f | MAE = %.4f | R2 = %.4f",
            test_metrics["mse"],
            test_metrics["mae"],
            test_metrics["r2"],
        )

        self.metadata = {
            "model_type": "Ridge",
            "best_params": best_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "data_info": {
                "train_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "total_samples": len(self.X),
                "feature_count": self.X.shape[-1],
            },
            "training_config": {
                "hyperparameter_tuning": bool(param_grid),
                "cv_folds": cv_folds if bool(param_grid) else 1,
            },
        }
        return self.model, self.metadata

    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Оценивает модель на заданных данных.

        Аргументы:
            X: Признаки.
            y: Целевая переменная.

        Вернёт:
            Метрики качества.
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        y_pred = self.model.predict(X)

        return {
            "mse": float(mean_squared_error(y, y_pred)),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
            "predictions_count": len(y_pred),
        }

    def save_model(self) -> tuple[str, str]:
        """Сохраняет модель и метаданные в директорию `resources_dir`.

        Вернёт:
            Пути к сохраненным файлам.
        """
        if self.model is None or self.metadata is None:
            raise ValueError("Модель или метаданные отсутствуют")

        os.makedirs(self.resources_dir, exist_ok=True)

        model_path = os.path.join(self.resources_dir, "model.pkl")
        metadata_path = os.path.join(self.resources_dir, "model_metadata.json")

        with open(model_path, "wb") as fin:
            pickle.dump(self.model, fin)

        with open(metadata_path, "w", encoding="utf-8") as fin:
            json.dump(self.metadata, fin, indent=4, ensure_ascii=False)

        self.logger.info("Модель сохранена: %s", model_path)
        self.logger.info("Метаданные сохранены: %s", metadata_path)

        return model_path, metadata_path
