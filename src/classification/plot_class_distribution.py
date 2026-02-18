import pandas as pd
import numpy as np
import warnings
import re
import logging
from typing import Tuple, List
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def plot_class_distribution(
    y_data_path: str = 'y_data.npy',
    figsize: tuple = (14, 8),
    bar_colors: list = None,
    pie_colors: list = None
) -> dict:
    """Визуализирует распределение классов в данных.

    Аргументы:
        y_data_path : Путь к файлу с метками классов.
        figsize : Размер фигуры (ширина, высота).
        bar_colors : Цвета для столбчатой диаграммы.
        pie_colors : Цвета для круговой диаграммы.

    Вернёт:
        Статистику по классам

    Исключения:
        FileNotFoundError: Если файл с данными не найден.
        ValueError: Если данные пустые.
    """
    # Загрузка данных
    y_data = np.load(y_data_path, allow_pickle=True)

    if len(y_data) == 0:
        raise ValueError("Данные пустые")

    # Подсчет количества каждого класса
    unique_classes, class_counts = np.unique(y_data, return_counts=True)
    total_samples = len(y_data)

    # Настройка цветов по умолчанию
    if bar_colors is None:
        bar_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    if pie_colors is None:
        pie_colors = ['#ff9999', '#66b3ff', '#99ff99']

    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. Столбчатая диаграмма
    bars = ax1.bar(
        unique_classes,
        class_counts,
        color=bar_colors[:len(unique_classes)]
    )

    ax1.set_xlabel('Уровень специалиста', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Количество резюме', fontsize=12, fontweight='bold')
    ax1.set_title(
        'Распределение резюме по уровням специалистов',
        fontsize=14,
        fontweight='bold'
    )

    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        percentage = count / total_samples * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.1,
            f'{count}\n({percentage:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # 2. Круговая диаграмма
    wedges, texts, autotexts = ax2.pie(
        class_counts,
        labels=unique_classes,
        autopct='%1.1f%%',
        colors=pie_colors[:len(unique_classes)],
        startangle=90,
        textprops={'fontsize': 11}
    )

    ax2.set_title(
        'Процентное соотношение уровней',
        fontsize=14,
        fontweight='bold'
    )

    for autotext in autotexts:
        autotext.set_fontweight('bold')

    fig.suptitle(
        f'Баланс классов | Всего резюме: {total_samples}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    # Статистика по балансу классов
    class_stats = {}
    for cls, count in zip(unique_classes, class_counts):
        percentage = count / total_samples * 100
        class_stats[cls] = {
            'count': int(count),
            'percentage': float(f'{percentage:.2f}')
        }

    # Проверка дисбаланса
    max_count = max(class_counts)
    min_count = min(class_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    class_stats['total_samples'] = total_samples
    class_stats['imbalance_ratio'] = float(f'{imbalance_ratio:.2f}')

    plt.tight_layout()
    plt.show()

    return class_stats

