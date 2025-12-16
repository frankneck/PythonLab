# -*- coding: utf-8 -*-

"""
Задача: выявление фальшивых банкнот (Banknote Authentication)
Метод: SVM (Support Vector Machine)
"""

import time
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------- Загрузка данных ----------------------

def load_banknote(path: str):
    """
    Загружает датасет Banknote Authentication.
    Ожидается 5 столбцов: 4 признака + класс (0 или 1)
    """
    try:
        data = np.loadtxt(path)
    except ValueError:
        data = np.loadtxt(path, delimiter=',')

    X = data[:, :4]   # признаки
    y = data[:, 4].astype(int)  # классы

    return X, y


# ---------------------- Фиксированное разбиение ----------------------

def split_fixed(X, y):
    """
    В тестовую выборку берём:
    - 107 последних объектов класса 0
    - 119 последних объектов класса 1
    """
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    test_idx = np.concatenate([
        idx_0[-107:],
        idx_1[-119:]
    ])

    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ---------------------- Базовые модели ----------------------

def run_baseline_models(X_train, y_train, X_test, y_test):
    """
    Проверка базовых настроек SVM
    """
    models = {
        "Linear SVM (C=1)": SVC(kernel="linear", C=1),
        "Poly SVM (deg=2)": SVC(kernel="poly", degree=2, C=1),
        "Poly SVM (deg=3)": SVC(kernel="poly", degree=3, C=1),
    }

    print("\nБазовые модели SVM\n")

    for name, clf in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", clf)
        ])

        start = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = pipe.predict(X_test)

        print(f"{name}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"Время обучения: {elapsed:.3f} сек\n")


# ---------------------- Кросс-валидация ----------------------

def run_grid_search(X_train, y_train, X_test, y_test):
    """
    Подбор оптимальных параметров SVM с помощью GridSearchCV
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC())
    ])

    param_grid = [
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10, 100]
        },
        {
            "svm__kernel": ["poly"],
            "svm__degree": [2, 3],
            "svm__C": [0.1, 1, 10],
            "svm__gamma": ["scale", "auto"]
        }
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    start = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - start

    print("\nЛучшие параметры (CV):")
    print(gs.best_params_)
    print(f"Лучшая CV-точность: {gs.best_score_:.4f}")
    print(f"Время подбора: {elapsed:.2f} сек")

    # Проверка на тесте
    y_pred = gs.predict(X_test)

    print("\nРезультаты на тестовой выборке:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

# ============================================================
# Эксперимент: AdultIncome — попытка улучшения точности SVM
# ============================================================

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def run_adult_income_experiment():
    """
    Эксперимент:
    Возможно ли улучшить точность SVM для AdultIncome
    за счёт подбора параметров C и gamma
    """

    print("\n" + "="*60)
    print("Эксперимент: AdultIncome + SVM")
    print("="*60)

    # ---------- Загрузка данных ----------
    adult = fetch_openml("adult", version=2, as_frame=True)
    X = adult.data
    y = (adult.target == ">50K").astype(int)

    # ---------- Признаки ----------
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # ---------- Train / Test ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---------- Базовая модель ----------
    baseline = Pipeline([
        ("prep", preprocess),
        ("svm", SVC(kernel="rbf", C=1, gamma="scale"))
    ])

    start = time.time()
    baseline.fit(X_train, y_train)
    base_time = time.time() - start

    y_pred_base = baseline.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred_base)

    print("\nБазовая модель SVM:")
    print(f"Accuracy: {base_acc:.4f}")
    print(f"Время обучения: {base_time:.2f} сек")

    # ---------- GridSearch ----------
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.1],
        "svm__kernel": ["rbf"]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    gs = GridSearchCV(
        baseline,
        param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    start = time.time()
    gs.fit(X_train, y_train)
    grid_time = time.time() - start

    y_pred_best = gs.predict(X_test)
    best_acc = accuracy_score(y_test, y_pred_best)

    print("\nПодбор параметров (GridSearch):")
    print("Лучшие параметры:", gs.best_params_)
    print(f"CV accuracy: {gs.best_score_:.4f}")
    print(f"Test accuracy: {best_acc:.4f}")
    print(f"Время подбора: {grid_time:.2f} сек")

    # ---------- Сравнение ----------
    print("\nСравнение результатов:")
    print(f"Baseline accuracy: {base_acc:.4f}")
    print(f"Best accuracy:     {best_acc:.4f}")
    print(f"Разница:           {best_acc - base_acc:+.4f}")


# ---------------------- Точка входа ----------------------

if __name__ == "__main__":
    X, y = load_banknote("SVM\data\data_banknote_authentication.txt")

    print("Размер данных:", X.shape)
    print("Распределение классов:", Counter(y))

    X_train, y_train, X_test, y_test = split_fixed(X, y)

    run_baseline_models(X_train, y_train, X_test, y_test)
    run_grid_search(X_train, y_train, X_test, y_test)
    
    # нахождение лучшего параметра C 
    run_adult_income_experiment()


