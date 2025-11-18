import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# finding outliners according to IQR method
def detectOutlierValueInCol_IQR(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low_border = Q1 - 1.5 * IQR
    high_border = Q3 + 1.5 * IQR
    outliers = df[(df[col] < low_border) | (df[col] > high_border)]
    print(f"{col}: Q1={Q1}, Q3={Q3}, IQR={IQR}, low={low_border}, high={high_border}, found={len(outliers)}")
    
    return outliers

def clearDataFrame(df):
    print(f"Исходный размер набора: {len(df)} строк")
    question_cols = [f"Q{i}" for i in range(1, 29)]

    # === 1. Аномалии с нулевой посещаемостью ===
    anomalies_attendance0 = df[df["attendance"] == 0]
    print(f"Нулевое посещение: {len(anomalies_attendance0)} записей")

    cleaned_df = df[df["attendance"] != 0].copy()

    # === 2. Аномалии, где все ответы одинаковые ===
    anomalies_same_answers = cleaned_df[cleaned_df[question_cols].nunique(axis=1) == 1]
    print(f"Все ответы одинаковые: {len(anomalies_same_answers)} записей")

    cleaned_df = cleaned_df[cleaned_df[question_cols].nunique(axis=1) > 1]

    # === 3. Посещаемость = 1 и есть хотя бы один высокий ответ (>= 4) ===
    df_attendance1 = cleaned_df[cleaned_df['attendance'] == 1]
    anomalies_high_with_low_attendance = df_attendance1[(df_attendance1[question_cols] >= 4).any(axis=1)]
    # print(f"Посещаемость = 1 и есть ответ >= 4: {len(anomalies_high_with_low_attendance)} записей")

    cleaned_df = cleaned_df.drop(anomalies_high_with_low_attendance.index)

    # === Большая повторяемость + низкие оценки ===
    df_repeat = cleaned_df[cleaned_df['nb.repeat'] >= 2]
    anomalies_suspicion_answers = df_repeat[(df_repeat[question_cols] == 1).any(axis=1)]

    # === 4. Собираем все аномалии в один DataFrame ===
    anomalies_all = pd.concat([
        anomalies_attendance0,
        anomalies_same_answers,
        anomalies_high_with_low_attendance,
        anomalies_suspicion_answers
    ]).drop_duplicates()

    # === 5. Статистика ===
    total_removed = len(anomalies_all)
    print(f"Всего удалено аномалий: {total_removed} строк")
    print(f"Итоговый размер набора: {len(cleaned_df)} строк")

    return cleaned_df, anomalies_all

def printHistOfCol(col):            
    print(df[col].hist(bins=20))
    plt.show()

def detectAllOutliers_IQR(df):
    IQR_outliers = pd.DataFrame()
    
    if "instr" in df: 
        df = df.sort_values(by="instr")
    
    for col in df.loc[:, "difficulty":"Q28"].columns:
        outliers = detectOutlierValueInCol_IQR(df, col)
        IQR_outliers = pd.concat([IQR_outliers, outliers], ignore_index=True)
    
    IQR_outliers = IQR_outliers.drop_duplicates()
    print(f'{IQR_outliers.head(10)}')
    print(f'Всего найдено выбросо с помощью IQR {len(IQR_outliers)}')

    return IQR_outliers

def detectOutliersCol_Z(df, col, threshold=2, distribution="uniform"):
    data = df[col].dropna()
    k = data.value_counts().sort_index()
    n = k.sum()
    
    if (distribution == "uniform"):
        p = 1 / len(k)
    elif (distribution == "empirical"):
        p = k / n

    freq = k / n
    sigma = np.sqrt(p * (1 - p) / n)
    z_score = (freq - p) / sigma
    outliers_values = z_score[abs(z_score) > threshold].index
    outliers_values = df[df[col].isin(outliers_values)]
    print(f"Z-score\n{z_score}")
    return outliers_values

def detectAllOutliers_Z(df):
    Z_Outliers = pd.DataFrame()

    if "instr" in df:
        df = df.sort_values(by="instr")

    for col in df.loc[:, "difficulty":"Q28"]:
        outliers = detectOutliersCol_Z(df, col, distribution="uniform")
        Z_Outliers = pd.concat([Z_Outliers, outliers])

    Z_Outliers = Z_Outliers.drop_duplicates()
    # print(f'{Z_Outliers}')
    print(f'Всего найдено выбросов с помощью Z-score = {len(Z_Outliers)}')
    
    return Z_Outliers

def common_matrix_cor(df):
    corr_matrix = df[['difficulty'] + [f'Q{i}' for i in range(1,29)]].corr()
    plt.figure(figsize=(16,14))
    sns.heatmap(corr_matrix, 
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                cbar=True,
                square=True)
    plt.title("Матрица кореляций", fontsize=18)
    plt.show()
    
def only_class_matrix_cor(df):
    os.makedirs("corr_matrices", exist_ok=True)
    cols_for_corr = ["difficulty"] + [f'Q{i}' for i in range(1,29)]

    for cls in df['class'].unique():
        df_class = df[df['class'] == cls]
        cor_matrix = df_class[cols_for_corr].corr()

        plt.figure(figsize=(16, 14))
        sns.heatmap(cor_matrix, 
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            square=True)
        
        plt.title(f'Correlation matrix for class {cls}')
        plt.tight_layout()
        plt.savefig(f'corr_matrices/class_{cls}.png')
        plt.close()

def detect_describing_stat_for_col(df, col, value_of_col):
    df_with_defined_col = df[df[col] == value_of_col]
    df_description_stat = df_with_defined_col.describe()
    
    group1 = [f'Q{i}' for i in range(1, 13)]
    group2 = [f'Q{i}' for i in range(13, 29)]
    
    common_mean_df = pd.DataFrame({
        'difficulty': [df_with_defined_col['difficulty'].mean()],
        'Q1-Q12': [df_with_defined_col[group1].mean().mean()],
        'Q13-Q28': [df_with_defined_col[group2].mean().mean()],
    })

    return common_mean_df, df_description_stat

def detect_describing_stat(df, col):
    instructors_list = sorted(df[col].unique())    
    
    common_mean_df_list = []
    common_description_df_list = []

    for i in range(1, len(instructors_list) + 1):
        common_mean_df, df_description_stat = detect_describing_stat_for_col(df, col, i)
        common_mean_df_list.append(common_mean_df)
        common_description_df_list.append(df_description_stat)
    
    return common_mean_df_list, common_description_df_list

def compareResults(df, col):
        # измени на instr, чтобы получить 3 df 
        list_mean, list_decr = detect_describing_stat(df, col)

        # === 1. Объединяем результаты в общую таблицу ===
        comparison_df = pd.concat(list_mean, ignore_index=True)
        comparison_df[col] = sorted(df[col].unique())
        comparison_df = comparison_df[[col,'difficulty','Q1-Q12','Q13-Q28']]

        # === 2. Вычисляем общий средний показатель преподавателя ===
        comparison_df["overall_mean"] = comparison_df[["Q1-Q12", "Q13-Q28"]].mean(axis=1)

        # === 3. Сортируем по общей средней (от лучшего к худшему) ===
        comparison_df = comparison_df.sort_values(by="overall_mean", ascending=False)

        # === 4. Выводим сводную таблицу сравнения ===
        if (col == "instr"):
            print("\n=== Сравнение преподавателей по средним показателям ===")
        elif (col == "class"):
            print("\n=== Сравнение предметов по средним показателям ===")
        else:
            print(f"\n=== Сравнение {col} по средним показателям ===")
        
        print(comparison_df.round(3))

        comparison_df.to_csv(f"{col}_instructors.csv", index=False)

        return 0

if __name__ == "__main__":
    df = pd.read_csv(r'Data\table.csv')
    
    # очистка df от аномалий + запись аномалий
    cleaned_df, anomalies = clearDataFrame(df)
    
    # нахождение выбросов с помощью IQR
    detectAllOutliers_IQR(cleaned_df)
    
    # нахождение выбросов с помощью Z-score
    outliers_z = detectAllOutliers_Z(cleaned_df)
    
    # общая матрица корреляций по средним показателям
    common_matrix_cor(cleaned_df)
    
    # сравнение предметов
    compareResults(cleaned_df, "class")

    # сравнение учителей по средним показателям
    compareResults(cleaned_df, "instr")
    
    # for i in range(1, 14):
    #     print(f"кол-во записей урока {i} равно {len(df.loc[df["class"] == i])}") # для 12-го урока всего 41 запись == отстойная матрица кореляции


    





