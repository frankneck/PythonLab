import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter


# Коллекции
label_names = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'
]


# --| Простой способ парсинга данных (уверены в соотношении) |--

# 1) Загрузка набора (формат: docID wordID count) 
def load_triplet_matrix(data_path, map_path, label_path):
    """
    Читает матрицу документ-терм из трёхколоночного формата:
    docId wordId count (индексы начиная с 1), как часто встречается.
    Возвращает: разреженную матрицу X(csr), матрицу меток классов y(np.array), словарь - массив всех слов из train+test vocab(list/np.array)
    """
    # labels
    y = np.loadtxt(label_path, dtype=int)
    # vocab / feature map (по-разному оформляют; парсим универсально)
    vocab = []
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                # допускаем форматы: "id<tab>token" ИЛИ просто "token"
                parts = line.split()
                if len(parts) == 1:
                    vocab.append(parts[0])
                else:
                    vocab.append(parts[-1])
    # data
    rows, cols, vals = [], [], []
    max_doc, max_word = 0, 0
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            d, w, c = line.split()
            d, w, c = int(d), int(w), float(c)
            rows.append(d-1)
            cols.append(w-1)
            vals.append(c)
            if d > max_doc: max_doc = d
            if w > max_word: max_word = w
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(max_doc, max_word), dtype=np.float64)
    # Если словарь пуст — сделаем фиктивные имена признаков
    if not vocab or len(vocab) != X.shape[1]:
        vocab = [f"term_{i+1}" for i in range(X.shape[1])]
    return X, y, np.array(vocab, dtype=object)

def load_dataset_matlab_folder(root="matlab"):
    train_data = os.path.join(root, "train.data")
    train_map  = os.path.join(root, "train.map")
    train_lbl  = os.path.join(root, "train.label")
    test_data  = os.path.join(root, "test.data")
    test_map   = os.path.join(root, "test.map")
    test_lbl   = os.path.join(root, "test.label")

    Xtr, ytr, vocab_tr = load_triplet_matrix(train_data, train_map, train_lbl)
    Xte, yte, vocab_te = load_triplet_matrix(test_data,  test_map,  test_lbl)

    # Проверка согласованности признаков
    if Xtr.shape[1] != Xte.shape[1]:
        raise ValueError(f"Число признаков не совпадает train={Xtr.shape[1]} vs test={Xte.shape[1]}")
    return Xtr, ytr, Xte, yte, vocab_tr


# --| Сложный способ парсинга данных (не уверены в соотношении) |--

# ---- NEW: чтение map-файла в список токенов (индекс = id-1) ----
def read_tokens_map(map_path):
    tokens = []
    with open(map_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # подхватываем форматы: "id token", "id\ttoken" или просто "token"
            parts = line.split()
            if len(parts) == 1:
                tok = parts[0]
            else:
                tok = parts[-1]
            tokens.append(tok)
    return tokens  # len = кол-во локальных термов, id = index+1

# ---- NEW: читаем triplets и конвертируем в (doc, global_term, count) через token ----
def read_triplets_remap_to_global(data_path, local_tokens, global_index):
    """
    local_tokens: список, где local_tokens[local_id-1] = token
    global_index: dict token -> global_id (0..|V|-1)
    Возвращает (rows, cols, vals, max_doc_id)
    """
    rows, cols, vals = [], [], []
    max_doc = 0
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            d, w, c = line.split()
            d = int(d)
            w = int(w)  # локальный id слова (1..len(local_tokens))
            c = float(c)
            if w < 1 or w > len(local_tokens):
                # пропустим некорректный id
                continue
            tok = local_tokens[w-1]
            gid = global_index.get(tok, None)
            if gid is None:
                # токена нет в глобальном словаре? (не должно быть, если глобальный строили как union)
                # на всякий случай — пропустим
                continue
            rows.append(d-1) # индекс строки = номер документа
            cols.append(gid)  # индекс столбца = глобальный id токена
            vals.append(c)  # значение = частота слова
            if d > max_doc: 
                max_doc = d # кол-во максимального номера документа
    return rows, cols, vals, max_doc

# ---- NEW: основной загрузчик с унификацией словаря ----
def load_dataset_matlab_folder_unified(root="matlab"):
    train_data = os.path.join(root, "train.data")
    train_map  = os.path.join(root, "train.map")
    train_lbl  = os.path.join(root, "train.label")
    test_data  = os.path.join(root, "test.data")
    test_map   = os.path.join(root, "test.map")
    test_lbl   = os.path.join(root, "test.label")

    # 1) читаем метки
    ytr = np.loadtxt(train_lbl, dtype=int)
    yte = np.loadtxt(test_lbl, dtype=int)

    # 2) читаем токены из обоих map
    tr_tokens = read_tokens_map(train_map)
    te_tokens = read_tokens_map(test_map)

    # 3) строим единый глобальный словарь по токенам (union, стабильный порядок)
    #    порядок: сначала все train-токены, затем те test-токены, которых нет в train
    global_tokens = list(tr_tokens)
    tr_set = set(tr_tokens)
    for tok in te_tokens:
        if tok not in tr_set:
            global_tokens.append(tok)

    global_index = {tok: i for i, tok in enumerate(global_tokens)}  # token -> global_id

    # 4) читаем triplets и немедленно ремапим в global_id
    tr_r, tr_c, tr_v, tr_maxdoc = read_triplets_remap_to_global(train_data, tr_tokens, global_index)
    te_r, te_c, te_v, te_maxdoc = read_triplets_remap_to_global(test_data,  te_tokens, global_index)

    # 5) строим CSR одинаковой ширины |V_global|
    V = len(global_tokens)
    Xtr = sparse.csr_matrix((tr_v, (tr_r, tr_c)), shape=(tr_maxdoc, V), dtype=np.float64)
    Xte = sparse.csr_matrix((te_v, (te_r, te_c)), shape=(te_maxdoc, V), dtype=np.float64)

    return Xtr, ytr, Xte, yte, np.array(global_tokens, dtype=object)



# 2) Подбор α ∈ (0,1) по CV
def tune_alpha(X, y, priors_mode="empirical", alphas=None, cv=5, random_state=42):
    """
    priors_mode: 'empirical' (fit_prior=True) или 'uniform' (fit_prior=False)
    alphas: список значений в (0,1)
    Возвращает: best_alpha, cv_table(pd.DataFrame)
    """
    if alphas is None:
        # логарифмическая сетка в (0,1): от 1e-4 до 1e-0 (не включая 1.0 ровно)
        # дополнительно несколько линейных точек
        a_log = np.geomspace(1e-4, 0.9, 12) # проверяем маленькие значения a
        a_lin = np.linspace(0.01, 0.99, 20) # линейная сетка для a = (0.01, 0.99)
        alphas = sorted(set([float(f"{a:.5f}") for a in np.concatenate([a_log, a_lin]) if 0 < a < 1])) # убираем дупликаты и сортируем set

    # кросс валидация
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state) # CV - части каждого дока. Берем каждую часть 
    rows = []
    for a in alphas:
        fold_scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            if priors_mode == "uniform":
                # uniform priors: fit_prior=False, class_prior=None
                clf = MultinomialNB(alpha=a, fit_prior=False) # сам классификатор
            else:
                # empirical priors (по долям в обучении): fit_prior=True
                clf = MultinomialNB(alpha=a, fit_prior=True) # сам классификатор
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_va)
            fold_scores.append(accuracy_score(y_va, pred))
        rows.append({"alpha": a, "mean_acc": float(np.mean(fold_scores)), "std_acc": float(np.std(fold_scores))})
    cv_table = pd.DataFrame(rows).sort_values("mean_acc", ascending=False).reset_index(drop=True)
    best_alpha = float(cv_table.iloc[0]["alpha"])
    return best_alpha, cv_table

# 3) Обучение двух моделей (равные априоры vs эмпирические)
def train_and_eval(Xtr, ytr, Xte, yte, alpha_emp, alpha_uni, target_names=None):
    # эмпирические априоры
    nb_emp = MultinomialNB(alpha=alpha_emp, fit_prior=True)
    nb_emp.fit(Xtr, ytr)
    pred_emp = nb_emp.predict(Xte)
    acc_emp = accuracy_score(yte, pred_emp)

    # равные априоры
    nb_uni = MultinomialNB(alpha=alpha_uni, fit_prior=False)
    nb_uni.fit(Xtr, ytr)
    pred_uni = nb_uni.predict(Xte)
    acc_uni = accuracy_score(yte, pred_uni)

    # классификационный отчет для каждой модели
    rep_emp = classification_report(yte, pred_emp, target_names=target_names, zero_division=0, output_dict=True)
    rep_uni = classification_report(yte, pred_uni, target_names=target_names, zero_division=0, output_dict=True)

    # матрицы ошибок для каждой модели
    cm_emp = confusion_matrix(yte, pred_emp)
    cm_uni = confusion_matrix(yte, pred_uni)

    return {
        "emp": {"alpha": alpha_emp, "acc": acc_emp, "report": rep_emp, "cm": cm_emp, "pred": pred_emp},
        "uni": {"alpha": alpha_uni, "acc": acc_uni, "report": rep_uni, "cm": cm_uni, "pred": pred_uni},
    }

# 4) Вспомогательные визуализации/сохранения
def save_results(results_dir, cv_emp, cv_uni, eval_dict, class_names):
    os.makedirs(results_dir, exist_ok=True)

    # CV таблицы
    cv_emp.to_csv(os.path.join(results_dir, "cv_curve_empirical.csv"), index=False)
    cv_uni.to_csv(os.path.join(results_dir, "cv_curve_uniform.csv"), index=False)

    # Графики валидации
    def plot_cv_curve(cv_tbl, title, fname):
        plt.figure()
        plt.plot(cv_tbl["alpha"], cv_tbl["mean_acc"], marker="o")
        plt.xlabel("alpha")
        plt.ylabel("CV accuracy")
        plt.title(title)
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, fname), dpi=200)
        plt.close()

    plot_cv_curve(cv_emp, "Validation curve (empirical class priors)", "cv_empirical.png")
    plot_cv_curve(cv_uni, "Validation curve (uniform class priors)", "cv_uniform.png")

    # Confusion matrices
    def plot_cm(cm, title, fname, labels):
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(np.arange(len(labels)), labels, rotation=90)
        plt.yticks(np.arange(len(labels)), labels)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, fname), dpi=220)
        plt.close()

    plot_cm(eval_dict["emp"]["cm"], f"Confusion Matrix (empirical, α={eval_dict['emp']['alpha']})",
            "cm_empirical.png", class_names)
    plot_cm(eval_dict["uni"]["cm"], f"Confusion Matrix (uniform, α={eval_dict['uni']['alpha']})",
            "cm_uniform.png", class_names)

    # Сводка метрик
    summary = {
        "empirical": {"alpha": eval_dict["emp"]["alpha"], "test_accuracy": eval_dict["emp"]["acc"]},
        "uniform":   {"alpha": eval_dict["uni"]["alpha"], "test_accuracy": eval_dict["uni"]["acc"]},
    }
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Полные классификационные отчёты (таблица)
    rep_emp_df = pd.DataFrame(eval_dict["emp"]["report"]).T
    rep_uni_df = pd.DataFrame(eval_dict["uni"]["report"]).T
    rep_emp_df.to_csv(os.path.join(results_dir, "classification_report_empirical.csv"))
    rep_uni_df.to_csv(os.path.join(results_dir, "classification_report_uniform.csv"))

def print_class_distribution(ytr, yte, label_names):
    print("\n=== Распределение документов по темам ===")
    counter_train = Counter(ytr)
    counter_test = Counter(yte)

    print(f"{'№ темы':<8}{'Название темы':<35}{'Train':>10}{'Test':>10}")
    print("-" * 65)
    for i, name in enumerate(label_names, start=1):
        train_count = counter_train.get(i, 0)
        test_count = counter_test.get(i, 0)
        print(f"{i:<8}{name:<35}{train_count:>10}{test_count:>10}")
    print("-" * 65)
    print(f"{'Итого':<43}{sum(counter_train.values()):>10}{sum(counter_test.values()):>10}")
    print("============================================\n")


def print_metrics(eval_dict, class_names):
    """
    eval_dict: словарь с результатами train_and_eval
    class_names: список названий классов
    """
    rows = []
    for model_name in ["emp", "uni"]:
        report = eval_dict[model_name]["report"]
        acc = eval_dict[model_name]["acc"]

        # Берём macro и weighted метрики
        row = {
            "Вариант модели": "Empirical" if model_name == "emp" else "Uniform",
            "Accuracy": acc,
            "Macro Precision": report["macro avg"]["precision"],
            "Macro Recall": report["macro avg"]["recall"],
            "Macro F1": report["macro avg"]["f1-score"],
            "Weighted Precision": report["weighted avg"]["precision"],
            "Weighted Recall": report["weighted avg"]["recall"],
            "Weighted F1": report["weighted avg"]["f1-score"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="{:.4f}".format))

# Пример использования после train_and_eval:
# 


# 5) main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="matlab", help="Папка с *.data/*.label/*.map")
    parser.add_argument("--results_dir", type=str, default="results", help="Куда сохранять результаты")
    parser.add_argument("--cv", type=int, default=5, help="Число фолдов для кросс-валидации")
    args = parser.parse_args()

    print("Загрузка данных...")
    Xtr, ytr, Xte, yte, vocab = load_dataset_matlab_folder_unified(args.data_dir)
    print_class_distribution(ytr, yte, label_names)
    print(f"Train shape: {Xtr.shape}, Test shape: {Xte.shape}, |V|={len(vocab)}")

    # Имена классов (если нет map для классов — используем индексы)
    # Часто в таких наборах метки уже 1..20 или 0..19; оставим как есть
    classes = np.unique(ytr)
    # Нормируем к 0..C-1 (на всякий случай)
    le = LabelEncoder()
    ytr = le.fit_transform(ytr)
    yte = le.transform(yte)
    class_names = [str(c) for c in le.classes_]

    # реальная частота класса в документах
    print("Подбор α (эмпирические априоры)...")
    best_alpha_emp, cv_emp = tune_alpha(Xtr, ytr, priors_mode="empirical", cv=args.cv)
    print("Лучшее alpha (empirical):", best_alpha_emp)

    # одинаковая для всех частота класса в документах
    print("Подбор α (равные априоры)...")
    best_alpha_uni, cv_uni = tune_alpha(Xtr, ytr, priors_mode="uniform", cv=args.cv)
    print("Лучшее alpha (uniform):", best_alpha_uni)

    # получаем 2 словаря (для модели с эмп априорами и равными)
    print("Финальное обучение и оценка на тесте...")
    eval_dict = train_and_eval(Xtr, ytr, Xte, yte, best_alpha_emp, best_alpha_uni, target_names=class_names)

    print(f"Test accuracy (empirical priors, α={eval_dict['emp']['alpha']}): {eval_dict['emp']['acc']:.4f}")
    print(f"Test accuracy (uniform priors,   α={eval_dict['uni']['alpha']}): {eval_dict['uni']['acc']:.4f}")

    save_results(args.results_dir, cv_emp, cv_uni, eval_dict, class_names)
    print(f"Готово. Смотрите папку '{args.results_dir}':")
    print("- cv_empirical.png / cv_uniform.png — кривые подбора α")
    print("- cm_empirical.png / cm_uniform.png — матрицы ошибок")
    print("- classification_report_*.csv — отчёты по классам (precision/recall/F1)")
    print("- summary.json — сводка по точности и выбранным α")

    print_metrics(eval_dict, class_names)

if __name__ == "__main__":
    main()


