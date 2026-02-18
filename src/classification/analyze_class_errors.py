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


def analyze_class_errors(y_test, y_pred, unique_classes):
    """Детальный анализ ошибок по классам.

    Аргументы:
        y_test: Истинные значения.
        y_pred: Предсказанные значения.
        unique_classes: Уникальные классы.
    """
    print("\n" + "="*60)
    print("АНАЛИЗ ОШИБОК ПО КЛАССАМ")
    print("="*60)

    for i, true_class in enumerate(unique_classes):
        idx = y_test == true_class
        correct = sum(y_pred[idx] == true_class)
        total = sum(idx)

        print(f"\nКласс {true_class}:")
        print(f"  Правильно классифицировано: {correct}/{total} ({correct/total*100:.1f}%)")

        # Какие ошибки чаще всего делаются
        pred_for_this_class = y_pred[idx]
        errors = pred_for_this_class[pred_for_this_class != true_class]

        if len(errors) > 0:
            error_counts = {cls: sum(errors == cls) for cls in unique_classes if cls != true_class}
            most_common_error = max(error_counts.items(), key=lambda x: x[1]) if error_counts else (None, 0)
            print(f"  Чаще всего путают с: {most_common_error[0]} ({most_common_error[1]} случаев)")

