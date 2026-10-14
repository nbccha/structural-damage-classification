"""Melatih dan mengevaluasi model LightGBM klasifikasi kerusakan bangunan."""

import json
import sys
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report
)
from lightgbm import LGBMClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess import (
    infer_target_column,
    detect_id_columns,
    identify_feature_types,
    build_preprocessor
)

from scripts.feature_engineering import engineer_features


def main():
    """Melatih model, mengevaluasi performa, dan menyimpan model final."""

    data_dir = PROJECT_ROOT / "data"

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    for path in [train_path, test_path, sample_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"File dataset tidak ditemukan: {path}"
            )

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    sample_submission = pd.read_csv(sample_path)

    target_column = infer_target_column(
        train_df,
        test_df,
        sample_submission
    )

    print("Target:", target_column)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    valid_mask = train_df[target_column].notna()

    y = train_df.loc[
        valid_mask,
        target_column
    ].reset_index(drop=True)

    X_raw = train_df.loc[valid_mask].drop(
        columns=[target_column]
    ).reset_index(drop=True)

    id_columns = detect_id_columns(X_raw)

    feature_columns = [
        col for col in X_raw.columns
        if col not in id_columns
    ]

    if not feature_columns:
        raise ValueError(
            "Tidak ada fitur yang tersedia untuk training."
        )

    missing_test = [
        col for col in feature_columns
        if col not in test_df.columns
    ]

    if missing_test:
        raise ValueError(
            f"Fitur training tidak tersedia pada test: {missing_test}"
        )

    X = engineer_features(
        X_raw[feature_columns],
        expected_columns=feature_columns
    )

    numerical_columns, categorical_columns = (
        identify_feature_types(X)
    )

    print("ID columns:", id_columns)
    print("Numerical features:", len(numerical_columns))
    print("Categorical features:", len(categorical_columns))

    stratify_target = (
        y if y.value_counts().min() >= 2 else None
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target
    )

    final_params = {
        "n_estimators": 500,
        "learning_rate": 0.08,
        "num_leaves": 63,
        "max_depth": 15,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_lambda": 0,
        "class_weight": "balanced"
    }

    validation_model = Pipeline([
        (
            "prep",
            build_preprocessor(
                numerical_columns,
                categorical_columns
            )
        ),
        (
            "algo",
            LGBMClassifier(
                objective="multiclass",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                **final_params
            )
        )
    ])

    validation_model.fit(X_train, y_train)

    y_pred = validation_model.predict(X_valid)

    validation_f1 = f1_score(
        y_valid,
        y_pred,
        average="macro",
        zero_division=0
    )

    validation_accuracy = accuracy_score(
        y_valid,
        y_pred
    )

    print(f"\nValidation Macro F1: {validation_f1:.6f}")
    print(f"Validation Accuracy: {validation_accuracy:.6f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_valid,
            y_pred,
            zero_division=0
        )
    )

    final_model = Pipeline([
        (
            "prep",
            build_preprocessor(
                numerical_columns,
                categorical_columns
            )
        ),
        (
            "algo",
            LGBMClassifier(
                objective="multiclass",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                **final_params
            )
        )
    ])

    final_model.fit(X, y)

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "final_damage_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    joblib.dump(final_model, model_path)

    metadata = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "id_columns": id_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "best_model_name": "LightGBM",
        "validation_macro_f1": float(validation_f1),
        "validation_accuracy": float(validation_accuracy),
        "model_params": final_params
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nTraining selesai.")
    print("Model disimpan:", model_path)
    print("Metadata disimpan:", metadata_path)


if __name__ == "__main__":
    main()