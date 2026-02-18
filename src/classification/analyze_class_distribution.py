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


def analyze_class_distribution(y):
    """Анализ распределения классов в данных.

    Аргументы:
        y: Вектор целевой переменной.

    Вернёт:
        unique_classes: Уникальные классы.
        class_counts: Количество образцов в каждом классе.
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)

    print("\nРаспределение классов:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  {cls}: {count} образцов ({count/len(y)*100:.1f}%)")

    return unique_classes, class_counts

