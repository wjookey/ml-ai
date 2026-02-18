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


def plot_confusion_matrix(y_test, y_pred, unique_classes, figsize=(10, 8)):
    """Визуализация матрицы ошибок.

    Аргументы:
        y_test: Истинные значения.
        y_pred: Предсказанные значения.
        unique_classes: Уникальные классы.
        figsize: Размер графика.
    """
    print("\n" + "="*60)
    print("МАТРИЦА ОШИБОК")
    print("="*60)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_classes,
                yticklabels=unique_classes)
    plt.title('Матрица ошибок классификации')
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.tight_layout()
    plt.show()

    return cm

