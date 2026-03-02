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


def prepare_data(X, y, test_size=0.3, random_state=42):
    """Подготовка данных: разделение на train/test и масштабирование.

    Аргументы:
        X: Матрица признаков.
        y: Вектор целевой переменной.
        test_size: Доля тестовых данных.
        random_state: Seed для воспроизводимости.

    Вернёт:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    print(f"\nРазмер train: {X_train.shape[0]}, размер test: {X_test.shape[0]}")

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler
