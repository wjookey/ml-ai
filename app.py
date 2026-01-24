import pandas as pd
from parsing_data import DataProcessingPipeline
import argparse
from pathlib import Path
import sys


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Обработка данных HH.ru для машинного обучения"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Путь к входному CSV файлу (например, hh.csv)",
    )

    return parser.parse_args()


def main():
    """Основная функция приложения"""
    args = parse_arguments()

    # Проверяем существование файла
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Ошибка: Файл {input_path} не найден")
        sys.exit(1)

    df = pd.read_csv(input_path, encoding="utf-8", index_col=0)
    df = df.drop_duplicates()
    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)

    # Создаем пайплайн и обрабатываем данные
    pipeline = DataProcessingPipeline()

    X, y = pipeline.process(df)
    pipeline.save(X, y)


if __name__ == "__main__":
    main()
