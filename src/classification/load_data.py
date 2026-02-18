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


def load_data(X_path='X_data.npy', y_path='y_data.npy'):
    """Загрузка данных из файлов .npy.

    Аргументы:
        X_path: Путь к файлу с признаками.
        y_path: Путь к файлу с целевой переменной.

    Вернёт:
        X: Матрица признаков.
        y: Вектор целевой переменной.
    """
    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)

    print(f"Размерность данных: X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"Количество признаков: {X.shape[1]}")

    return X, y

