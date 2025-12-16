# file: run_nn_mse_kc_with_dfhandler.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from DataFrameHandler import DataFrameHandler

# ====== настройки ======
DATA_PATH    = "Neuro\kc_house_clean.csv"
TARGET_COL   = "price"
TEST_SIZE    = 0.3
RANDOM_STATE = 42
OUTDIR       = "results_nn"
FEATURE_RANGE = (0.0, 1.0)

# Конфигурации MLP
MLP_CONFIGS = [
    dict(name="mlp_cfg1", hidden_layer_sizes=(16,), activation="relu", solver="adam", max_iter=600, alpha=1e-4),
    dict(name="mlp_cfg2", hidden_layer_sizes=(32,16), activation="relu", solver="adam", max_iter=3000, alpha=5e-4),
    dict(name="mlp_cfg3", hidden_layer_sizes=(32,), activation="tanh", solver="adam", max_iter=800, alpha=1e-4),
]

def plot_pred_vs_true(y_true, y_pred, title, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=10)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo,hi],[lo,hi], 'r--')
    plt.xlabel("Истинная цена")
    plt.ylabel("Прогноз")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) Загрузка
    df = pd.read_csv(DATA_PATH)

    # 2) Очистка и заполнение пропусков через DataFrameHandler
    df = DataFrameHandler.FillMissingValues(df)
    df = DataFrameHandler.DetectOutliersIQR(df)
    df = DataFrameHandler.Standartize(df, skip_cols=[TARGET_COL])
    df = DataFrameHandler.FeatureExtraction(df)
    df = DataFrameHandler.Standartize(df, skip_cols=[TARGET_COL])

    # 3) Подготовка данных
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # 4) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5) Масштабирование [0,1]
    preproc = ColumnTransformer(
        transformers=[("mm", MinMaxScaler(feature_range=FEATURE_RANGE), list(range(X_train.shape[1])))],
        remainder="drop"
    )

    results = []

    # 6) Обучение нескольких конфигураций MLP
    for cfg in MLP_CONFIGS:
        name = cfg["name"]
        mlp = MLPRegressor(
            random_state=RANDOM_STATE,
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation=cfg["activation"],
            solver=cfg["solver"],
            max_iter=cfg["max_iter"],
            alpha=cfg["alpha"]
        )

        pipe = Pipeline([("scale", preproc), ("mlp", mlp)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mean_y = np.mean(y_test)
        relative_error = rmse / mean_y * 100

        # Сохранение предсказаний
        pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
            os.path.join(OUTDIR, f"preds_{name}.csv"), index=False, encoding="utf-8"
        )

        # График предсказанное vs реальное
        plot_pred_vs_true(y_test, y_pred,
                          title=f"{name} (СКО={rmse:.1f})",
                          outpath=os.path.join(OUTDIR, f"plot_pred_vs_true_{name}.png"))

        results.append({"model": f"Нейронная сеть ({name})", "rmse": rmse, "relative_error": relative_error})

    # 7) Итоговая таблица
    res_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    res_df.to_csv(os.path.join(OUTDIR, "metrics_rmse.csv"), index=False, encoding="utf-8")

    print("="*70)
    print("Итоговая таблица (СКО, меньше — лучше):")
    print(res_df.to_string(index=False))
    print(f"Сохранено: {os.path.join(OUTDIR, 'metrics_rmse.csv')}")
    print("Готово.")

if __name__ == "__main__":
    main()
