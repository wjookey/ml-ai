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


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model=None, cv_folds=5):
    """Обучение модели с кросс-валидацией и оценка на тестовых данных.

    Аргументы:
        X_train: Обучающие признаки.
        y_train: Обучающая целевая переменная.
        X_test: Тестовые признаки.
        y_test: Тестовая целевая переменная.
        model: Модель для обучения (по умолчанию LogisticRegression).
        cv_folds: Количество фолдов для кросс-валидации.

    Вернёт:
        model: Обученная модель.
        accuracy: Точность на тестовых данных.
        cv_scores: Результаты кросс-валидации.
    """
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=42)

    print("\n" + "="*60)
    print("ОЦЕНКА МОДЕЛЕЙ (кросс-валидация, {} фолдов)".format(cv_folds))
    print("="*60)

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  Средняя точность кросс-валидации: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"  Точность на тестовых данных: {accuracy:.3f}")

    return model, accuracy, cv_scores

