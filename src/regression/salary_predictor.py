from utils import setup_logger
import numpy as np
from sklearn.linear_model import Ridge
from typing import Any
import os
import pickle
import json


class SalaryPredictor:
    """Класс для загрузки обученной модели и предсказания зарплат."""

    def __init__(self, resources_dir: str = "resources") -> None:
        """Инициализация SalaryPredictor.

        Аргументы:
            resources_dir: Директория с сохранённой моделью и метаданными.
        """
        self.resources_dir = resources_dir
        self.logger = setup_logger(self.__class__.__name__)

        self.model: Ridge | None = None
        self.metadata: dict[str, Any] | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Загружает сохранённую модель и метаданные из `resources_dir`.

        Исключения:
            FileNotFoundError: Если файл модели не найден.
        """
        model_path = os.path.join(self.resources_dir, "model.pkl")
        metadata_path = os.path.join(self.resources_dir, "model_metadata.json")

        if not os.path.exists(model_path):
            self.logger.error("Файл модели не найден: %s", model_path)
            raise FileNotFoundError(
                f"Модель не найдена в {model_path}. Сначала обучите модель."
            )

        with open(model_path, "rb") as fin:
            self.model = pickle.load(fin)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as fin:
                self.metadata = json.load(fin)

        self.logger.info("Модель загружена из %s", model_path)

        if self.metadata:
            self.logger.info(
                "Тип модели: %s",
                self.metadata.get("model_type", "Unknown"),
            )

    def predict(self, X_data: str) -> list[float]:
        """Предсказывает зарплаты по входным признакам.

        Аргументы:
            X_data: Массив признаков (`np.ndarray`) или путь к файлу с признаками.

        Вернёт:
            Предсказанные значения зарплат.

        Исключения:
            ValueError: Если модель не загружена.
            FileNotFoundError: Если файл с признаками не найден.
        """
        if self.model is None:
            raise ValueError("Модель не загружена")

        if isinstance(X_data, str):
            if not os.path.exists(X_data):
                raise FileNotFoundError(f"Файл с признаками не найден: {X_data}")
            X_data = np.load(X_data, allow_pickle=True)

        self.logger.debug("Начато предсказание | data.shape = %s", X_data.shape)

        predictions = self.model.predict(X_data)
        return [float(value) for value in predictions]
