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


def calculate_baseline(y_test, class_counts, unique_classes):
    """Расчет baseline модели (предсказание самого частого класса).

    Аргументы:
        y_test: Тестовая целевая переменная.
        class_counts: Количество образцов в каждом классе.
        unique_classes: Уникальные классы.

    Вернёт:
        baseline_accuracy: Точность baseline модели.
    """
    most_common_class = unique_classes[np.argmax(class_counts)]
    baseline_accuracy = sum(y_test == most_common_class) / len(y_test)

    print(f"\n" + "="*60)
    print(f"Точность baseline (всегда предсказываем '{most_common_class}'): {baseline_accuracy:.3f}")
    print("="*60)

    return baseline_accuracy, most_common_class

