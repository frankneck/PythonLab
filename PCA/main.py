import os
import sys
import pandas as pd
from PcaHandler import PcaHandler
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Linear_regression", "Source"))

# Now you can import your class
from DataFrameHandler import DataFrameHandler

COL = 'attendance'
ATTENDANCE_VALUE = 1
HIGH_ANSWER_VALUE = 5
COLS = [f'Q{i}' for i in range(1, 29)]
SKIPPED_COLS = ['id', 'nb.repeat', 'attendance', 'instr', 'class', 'difficulty']


def clear_dataframe(df) :
    print(f"Было: {df.shape[0]}")
    total_removed = 0
    df, df_removed_step = DataFrameHandler.clear_by_col_value(df, COL, 0)
    print(f"Удалено записей с нулевым посещением: {df_removed_step.shape[0]}")
    total_removed += df_removed_step.shape[0]

    df, df_removed_step = DataFrameHandler.clear_same_cols(df, [f'Q{i}' for i in range(1, 29)])
    total_removed += df_removed_step.shape[0]
    print(f"Удалено записей с одинаковыми ответами: {df_removed_step.shape[0]}")

    df, df_removed_step = DataFrameHandler.clear_different_col_value_between_cols(
        df, COL, ATTENDANCE_VALUE, COLS, HIGH_ANSWER_VALUE, "<="
    )
    print(f"Удалено записей с посещением 1 и ответом {HIGH_ANSWER_VALUE}: {df_removed_step.shape[0]}")
    total_removed += df_removed_step.shape[0]

    print(f"Очистка завершена. Удалено всего записей: {total_removed}")
    print(f"Стало: {df.shape[0]}\n")

    return df

def Execute_PCA_Analyses(df, suffix=""):
    PcaHandler.df_pca_analysis(df, class_name=1, skipped_cols=SKIPPED_COLS, title_suffix=suffix)
    PcaHandler.df_pca_two_classes_by_teacher(df, teacher_id=1, skipped_cols=SKIPPED_COLS, title_suffix=suffix)
    PcaHandler.df_pca_all(df, skipped_cols=SKIPPED_COLS, title_suffix=suffix)

if __name__ == "__main__":
    df = pd.read_csv(r'Data\table.csv')
    df = clear_dataframe(df)
    
    df_stand = DataFrameHandler.standartize(df, SKIPPED_COLS).copy()
    df_not_stand = df.copy()

    # print("=== PCA с стандартизацией ===")
    # Execute_PCA_Analyses(df_stand, suffix="_stand")
    print("=== PCA без стандартизации ===")
    Execute_PCA_Analyses(df_not_stand, suffix="_not_stand")
    
