import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
)

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _encode_features(features) -> pd.DataFrame:
    df = pd.DataFrame(features).copy()
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _filter_iqr_outliers(
    X_df: pd.DataFrame, y_ser: pd.Series, is_classification: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    numeric_cols = [col for col in X_df.columns if X_df[col].nunique() > 2]
    if not numeric_cols:
        return X_df.to_numpy(dtype=float), y_ser.to_numpy(dtype=float)

    q1 = X_df[numeric_cols].quantile(0.25)
    q3 = X_df[numeric_cols].quantile(0.75)
    iqr = q3 - q1

    valid_cols = iqr[iqr > 0].index.tolist()
    if not valid_cols:
        return X_df.to_numpy(dtype=float), y_ser.to_numpy(dtype=float)

    lower_bound = q1[valid_cols] - 1.5 * iqr[valid_cols]
    upper_bound = q3[valid_cols] + 1.5 * iqr[valid_cols]

    is_outlier = (X_df[valid_cols] < lower_bound) | (X_df[valid_cols] > upper_bound)
    keep_mask = ~is_outlier.any(axis=1)

    if keep_mask.sum() < 10:
        return X_df.to_numpy(dtype=float), y_ser.to_numpy(dtype=float)
    if is_classification and y_ser[keep_mask].nunique() < y_ser.nunique():
        return X_df.to_numpy(dtype=float), y_ser.to_numpy(dtype=float)

    return X_df[keep_mask].to_numpy(dtype=float), y_ser[keep_mask].to_numpy(dtype=float)


def get_and_process_student_data(path_to_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv)
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    temp_df = df.drop(columns=["Extracurricular Activities"])
    q1 = temp_df.quantile(0.25)
    q3 = temp_df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    is_outlier = (temp_df < lower_bound) | (temp_df > upper_bound)
    clean_df = df[~is_outlier.any(axis=1)]
    return clean_df


def get_and_process_german_data(path_to_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv, sep=";")
    feature_cols = [c for c in df.columns if c != "Creditability" and df[c].nunique() > 2]
    q1 = df[feature_cols].quantile(0.25)
    q3 = df[feature_cols].quantile(0.75)
    iqr = q3 - q1
    valid_cols = iqr[iqr > 0].index.tolist()
    lower_bound = q1[valid_cols] - 1.5 * iqr[valid_cols]
    upper_bound = q3[valid_cols] + 1.5 * iqr[valid_cols]
    is_outlier = (df[valid_cols] < lower_bound) | (df[valid_cols] > upper_bound)
    clean_df = df[~is_outlier.any(axis=1)]
    return clean_df


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train_df = _encode_features(data.train_features)
    y_train_ser = pd.Series(np.asarray(data.train_target, dtype=float))
    X, Y = _filter_iqr_outliers(X_train_df, y_train_ser, is_classification=False)

    X_test_raw = _encode_features(data.test_features).to_numpy(dtype=float)

    X_means = X.mean(axis=0)
    X_stds = X.std(axis=0)
    X_stds = np.where(X_stds == 0, 1.0, X_stds)
    X_train = (X - X_means) / X_stds

    w = np.zeros(X_train.shape[1])
    b = 0.0
    loss_history = []

    for _ in range(data.epochs):
        error = Y - (X_train @ w + b)
        loss_history.append(float((error ** 2).mean()))
        w = w + (data.learning_rate * 2 * (X_train.T @ error) / X_train.shape[0])
        b = b + float(data.learning_rate * 2 * error.mean())

    X_test_norm = (X_test_raw - X_means) / X_stds
    predictions = X_test_norm @ w + b

    return RegressionResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train_df = _encode_features(data.train_features)
    y_train_ser = pd.Series(np.asarray(data.train_target, dtype=float))
    X, Y = _filter_iqr_outliers(X_train_df, y_train_ser, is_classification=True)

    X_test_raw = _encode_features(data.test_features).to_numpy(dtype=float)

    X_means = X.mean(axis=0)
    X_stds = X.std(axis=0)
    X_stds = np.where(X_stds == 0, 1.0, X_stds)
    X_train = (X - X_means) / X_stds

    w = np.zeros(X_train.shape[1])
    b = 0.0
    loss_history = []

    for _ in range(data.epochs):
        z = np.clip(X_train @ w + b, -500, 500)
        p = 1 / (1 + np.exp(-z))
        p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
        loss = Y * np.log(p_clipped) + (1 - Y) * np.log(1 - p_clipped)
        loss_history.append(float((-loss).mean()))

        error = Y - p
        w = w + (data.learning_rate * (X_train.T @ error) / X_train.shape[0])
        b = b + float(data.learning_rate * error.mean())

    X_test_norm = (X_test_raw - X_means) / X_stds
    probabilities = 1 / (1 + np.exp(-np.clip(X_test_norm @ w + b, -500, 500)))
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    X_train_df = _encode_features(data.train_features)
    y_train_ser = pd.Series(np.asarray(data.train_target, dtype=float))
    X_train_raw, y_train_flt = _filter_iqr_outliers(X_train_df, y_train_ser, is_classification=True)
    y_train = y_train_flt.astype(int)

    X_test_raw = _encode_features(data.test_features).to_numpy(dtype=float)
    y_test = np.asarray(data.test_target, dtype=int)

    X_means = X_train_raw.mean(axis=0)
    X_stds = X_train_raw.std(axis=0)
    X_stds = np.where(X_stds == 0, 1.0, X_stds)

    X_train = (X_train_raw - X_means) / X_stds
    X_test = (X_test_raw - X_means) / X_stds

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    dt_model = DecisionTreeClassifier(random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)

    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]

    dt_pred = dt_model.predict(X_test)
    dt_prob = dt_model.predict_proba(X_test)[:, 1]

    knn_pred = knn_model.predict(X_test)
    knn_prob = knn_model.predict_proba(X_test)[:, 1]

    return BaselineComparisonResult(
        logistic_regression=calculate_classification_metrics(y_test, lr_pred, lr_prob),
        decision_tree=calculate_classification_metrics(y_test, dt_pred, dt_prob),
        knn=calculate_classification_metrics(y_test, knn_pred, knn_prob),
    )


def metrics_to_dict(m: ClassificationMetrics) -> dict[str, float]:
    return {
        "roc_auc": m.roc_auc,
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
    }


if __name__ == "__main__":
    df_student = (
        get_and_process_student_data("Student_Performance.csv")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    train_size_reg = int(0.8 * len(df_student))

    reg_input = RegressionInput(
        train_features=df_student[:train_size_reg].drop(columns=["Performance Index"]),
        train_target=df_student[:train_size_reg]["Performance Index"],
        test_features=df_student[train_size_reg:].drop(columns=["Performance Index"]),
        test_target=df_student[train_size_reg:]["Performance Index"],
    )
    reg_result = train_linear_regression(reg_input)
    test_mse = float(mean_squared_error(reg_input.test_target, reg_result.predictions))
    print(f"Linear MSE: {test_mse:.4f}")

    german_data = (
        get_and_process_german_data("german.csv")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    train_size_clf = int(0.8 * len(german_data))

    clf_input = ClassificationInput(
        train_features=german_data[:train_size_clf].drop(columns=["Creditability"]),
        train_target=german_data[:train_size_clf]["Creditability"],
        test_features=german_data[train_size_clf:].drop(columns=["Creditability"]),
        test_target=german_data[train_size_clf:]["Creditability"],
    )
    own_clf_result = train_logistic_classifier(clf_input)
    own_metrics = calculate_classification_metrics(
        np.asarray(clf_input.test_target, dtype=int),
        own_clf_result.predictions,
        own_clf_result.probabilities,
    )

    comparison_result = compare_classical_models(clf_input)

    experiment_results = {
        "linear_regression": {
            "test_mse": test_mse
        },
        "classification": {
            "own_logistic": metrics_to_dict(own_metrics),
            "logistic_regression": metrics_to_dict(comparison_result.logistic_regression),
            "decision_tree": metrics_to_dict(comparison_result.decision_tree),
            "knn": metrics_to_dict(comparison_result.knn),
        },
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    print(json.dumps(experiment_results, indent=2))