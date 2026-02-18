# Структура проекта

Корневая структура репозитория:

- `src/` — исходный код
  - `classification/` — скрипты и данные для задач классификации
  - `regression/` — скрипты и данные для задач регрессии
  - `parsing/` — утилиты для разбора и подготовки данных

Файлы в `src/classification`:
- `app.py` — основной скрипт
- `load_data.py` — загрузка данных
- `plot_class_distribution.py` — построение графика распределения классов
- `analyze_class_distribution.py` — анализ классового распределеия
- `prepare_data.py` — подготовка и предобработка
- `train_and_evaluate_model.py` — обучение и оценка
- `generate_classification_report.py` — генерация отчета классификации
- `plot_confusion_matrix.py` — визуализация матрицы ошибок
- `analyze_class_errors.py` — анализ ошибок по классам
- `calculate_baseline.py` — расчёт baseline модели
- `X_data.npy`, `y_data.npy` — примеры входных данных

Файлы в `src/regression`:
- `app.py` — основной скрипт
- `utils.py` — функция логирования
- `model_trainer.py` — класс для обучения, оценки и сохранения модели Ridge
- `salary_predictor.py` — класс для загрузки обученной модели и предсказания зарплат
- `X_data.npy`, `y_data.npy` — примеры входных данных
- `predicted_salaries.csv` — предсказанная зарплата

Файлы в `src/parsing`:
- `app.py` — основной скрипт
- `parsing_data.py` — пайплайн по парсингу датасета
- `X_data.npy`, `y_data.npy` — полученные данные

Документация разделена по главам в папке `docs/`.
