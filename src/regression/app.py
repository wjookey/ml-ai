import pandas as pd
import argparse
from pathlib import Path
import warnings
from typing import Any
from model_trainer import ModelTrainer
from salary_predictor import SalaryPredictor
import numpy as np

warnings.filterwarnings("ignore")


RIDGE_PARAM_GRID: dict[str, list[Any]] = {
    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    "fit_intercept": [True, False],
    "solver": ["auto", "svd", "cholesky"],
}


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Обучение модели линейной регрессии"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Путь к директории с X_data.npy и y_data.npy",
    )

    return parser.parse_args()


def main():
    """Основная функция приложения."""
    args = parse_arguments()
    input_path = Path(args.input_path)
    res_dir = input_path / "resources"
    trainer = ModelTrainer(data_dir=input_path, resources_dir=res_dir)
    trainer.load_data()
    trainer.split_data()
    trainer.train_model(param_grid=RIDGE_PARAM_GRID, cv_folds=5)
    trainer.save_model()

    x_data_path = input_path / "X_data.npy"
    predictor = SalaryPredictor(res_dir)
    salaries = predictor.predict(str(x_data_path))
    rounded_salaries = np.round(salaries, 2)

    output_path = Path("predicted_salaries.csv")
    pd.DataFrame({"salary": rounded_salaries}).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
