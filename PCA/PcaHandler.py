from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import os

class PcaHandler:

    data_dir = "Data"
    sheets_dir = os.path.join(data_dir, "sheets")
    graphics_dir = os.path.join(data_dir, "graphics")

    # Создаём папки, если их нет
    os.makedirs(sheets_dir, exist_ok=True)
    os.makedirs(graphics_dir, exist_ok=True)

    @staticmethod
    def run_pca(X, title="PCA", title_suffix=""):
        """
        Выполняет PCA, строит графики, сохраняет результаты
        + график накопленной дисперсии
        + пунктирные линии 90%, 95%, 99%
        + вывод количества компонент для достижения порогов.
        """

        # PCA
        pca = PCA()
        X_pca_array = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        # DataFrame проекции
        pc_columns = [f"PC{i+1}" for i in range(X_pca_array.shape[1])]
        X_pca = pd.DataFrame(X_pca_array, columns=pc_columns, index=X.index)

        # Loadings
        loadings = pd.DataFrame(pca.components_, columns=X.columns)

        # Безопасное имя файла
        title_safe = title.replace(" ", "_") + title_suffix

        # --- Сохраняем таблицы ---
        X_pca.to_csv(f"{PcaHandler.sheets_dir}/{title_safe}_projection.csv", index=False)
        loadings.to_csv(f"{PcaHandler.sheets_dir}/{title_safe}_loadings.csv", index=False)

        # --- График накопленной доли дисперсии ---
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative) + 1), cumulative, marker='o')

        # Пороговые уровни
        thresholds = [0.90, 0.95, 0.99]
        for thr in thresholds:
            plt.axhline(y=thr, color='gray', linestyle='--')
            plt.text(1, thr + 0.01, f"{int(thr*100)}%", color='gray')

        plt.xlabel("Номер компоненты")
        plt.ylabel("Накопленная доля дисперсии")
        plt.title(f"{title} - Накопленная доля объяснённой дисперсии")
        plt.grid(True)

        plt.savefig(f"{PcaHandler.graphics_dir}/{title_safe}_cumulative_variance.png")
        plt.close()

        # --- Анализ PC1 ---
        print(f"\n=== {title} ===")
        print(f"Главная компонента (PC1) объясняет {explained[0]:.2f} доли дисперсии")

        # --- Топ-10 признаков PC1 ---
        pc1_loadings = loadings.iloc[0]
        top10 = pc1_loadings.abs().sort_values(ascending=False).head(10)

        print("Топ-10 признаков (loading) по модулю:")
        for feat, val in top10.items():
            print(f"  {feat}: {val:.4f}")

        top_features = pd.DataFrame(top10).reset_index()
        top_features.columns = ['Feature', 'Loading']
        top_features_file = f"{PcaHandler.sheets_dir}/{title_safe}_top10_features.csv"
        top_features.to_csv(top_features_file, index=False)
        print(f"Топ-10 признаков сохранены в {top_features_file}")

        # --- Сколько компонент нужно для достижения порогов ---
        print("\n--- Требуемое число компонент ---")
        for thr in thresholds:
            k = np.argmax(cumulative >= thr) + 1
            print(f"Для {int(thr*100)}%: нужно компонент = {k}")

        return X_pca, loadings, explained


    @staticmethod
    def df_pca_analysis(df, class_name, skipped_cols, title_suffix=""):
        """PCA для одного класса"""
        df_class = df[df['class'] == class_name]
        if df_class.empty:
            print(f"Класс {class_name}: нет данных!")
            return None, None, None
        cols = [c for c in df.columns if c not in skipped_cols + ['class']]
        X = df_class[cols]
        print(f"Объём данных для класса {class_name}: {len(X)}")
        title = f"PCA для класса {class_name}"
        return PcaHandler.run_pca(X, title=title, title_suffix=title_suffix)

    @staticmethod
    def df_pca_two_classes_by_teacher(df, teacher_id, skipped_cols, title_suffix=""):
        """PCA для двух классов одного преподавателя"""
        df_teacher = df[df['instr'] == teacher_id]
        if df_teacher.empty:
            print(f"Преподаватель {teacher_id}: нет данных!")
            return None, None, None

        classes = df_teacher['class'].unique()
        if len(classes) < 2:
            print(f"Преподаватель {teacher_id} имеет меньше двух классов, берём все имеющиеся")
        selected_classes = classes[:2]
        df_two = df_teacher[df_teacher['class'].isin(selected_classes)]

        cols = [c for c in df.columns if c not in skipped_cols + ['instr', 'class']]
        X = df_two[cols]
        print(f"Объём данных для преподавателя {teacher_id}, классы {selected_classes}: {len(X)}")
        title = f"PCA для преподавателя {teacher_id}, классы {selected_classes}"
        return PcaHandler.run_pca(X, title=title, title_suffix=title_suffix)

    @staticmethod
    def df_pca_all(df, skipped_cols, title_suffix=""):
        """PCA для всего набора данных"""
        cols = [c for c in df.columns if c not in skipped_cols + ['class']]
        X = df[cols]
        print(f"Объём всех данных: {len(X)}")
        title = "PCA для всех данных"
        return PcaHandler.run_pca(X, title=title, title_suffix=title_suffix)
