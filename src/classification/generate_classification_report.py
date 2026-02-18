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


def generate_classification_report(model, X_test, y_test, unique_classes):
    """Генерация отчета о классификации.

    Аргументы:
        model: Обученная модель.
        X_test: Тестовые признаки.
        y_test: Тестовая целевая переменная.
        unique_classes: Уникальные классы.

    Вернёт:
        y_pred: Предсказанные значения.
        report: Отчет о классификации.
    """
    print("\n" + "="*60)
    print("ОТЧЕТ О КЛАССИФИКАЦИИ")
    print("="*60)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=unique_classes)
    print(report)

    return y_pred, report

