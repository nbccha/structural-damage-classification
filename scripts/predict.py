"""Prediksi data test dan pembuatan submission Kaggle."""

import json
import re
import sys
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from pandas.api.types import is_numeric_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.feature_engineering import engineer_features


def convert_predictions(predictions, sample_target):
    """
    Menyesuaikan tipe label prediksi dengan template submission.

    Parameters
    ----------
    predictions : array-like
        Hasil prediksi model.
    sample_target : pandas.Series
        Kolom target dari template submission.

    Returns
    -------
    pandas.Series
        Prediksi dengan tipe label yang sesuai template.
    """

    predictions = pd.Series(predictions)

    if is_numeric_dtype(sample_target.dtype):
        numeric_predictions = pd.to_numeric(
            predictions,
            errors="coerce"
        )

        # Mengonversi label seperti 'Grade 1' menjadi angka 1.
        if numeric_predictions.isna().any():
            extracted = predictions.astype(str).str.extract(
                r"(\d+)",
                expand=False
            )

            numeric_predictions = pd.to_numeric(
                extracted,
                errors="coerce"
            )

        if numeric_predictions.isna().any():
            raise ValueError(
                "Label prediksi tidak dapat dikonversi "
                "ke format numerik."
            )

        if not np.allclose(
            numeric_predictions,
            np.round(numeric_predictions)
        ):
            raise ValueError(
                "Label numerik prediksi bukan bilangan bulat."
            )

        return numeric_predictions.astype(int)

    return predictions


def main():
    """Memuat model, memprediksi data test, dan menyimpan submission."""

    # 1. Menentukan lokasi file.
    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "models"
    submission_dir = PROJECT_ROOT / "submissions"

    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    model_path = model_dir / "final_damage_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    required_files = [
        test_path,
        sample_path,
        model_path,
        metadata_path
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {path}. "
                "Pastikan scripts/train.py sudah dijalankan."
            )

    # 2. Memuat dataset, model, dan metadata.
    test_df = pd.read_csv(test_path, low_memory=False)
    sample_submission = pd.read_csv(sample_path)
    final_model = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    target_column = metadata["target_column"]
    feature_columns = metadata["feature_columns"]
    id_columns = metadata.get("id_columns", [])

    if target_column not in sample_submission.columns:
        raise ValueError(
            f"Kolom target '{target_column}' tidak ditemukan "
            "dalam sample_submission.csv."
        )

    if len(test_df) != len(sample_submission):
        raise ValueError(
            "Jumlah baris test tidak sama dengan template submission."
        )

    print("Test shape:", test_df.shape)
    print("Model:", metadata["best_model_name"])

    # 3. Menyesuaikan urutan data dengan template melalui ID.
    shared_ids = [
        col for col in id_columns
        if col in test_df.columns
        and col in sample_submission.columns
        and col != target_column
    ]

    if not shared_ids:
        shared_ids = [
            col for col in sample_submission.columns
            if col != target_column
            and col in test_df.columns
            and test_df[col].is_unique
            and sample_submission[col].is_unique
        ]

    if shared_ids:
        if test_df.duplicated(subset=shared_ids).any():
            raise ValueError("ID pada test tidak unik.")

        if sample_submission.duplicated(
            subset=shared_ids
        ).any():
            raise ValueError("ID pada template tidak unik.")

        ordered_test = sample_submission[
            shared_ids
        ].merge(
            test_df,
            on=shared_ids,
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True
        )

        if (ordered_test["_merge"] != "both").any():
            raise ValueError(
                "Ada ID pada template yang tidak ditemukan pada test."
            )

        test_df = ordered_test.drop(columns="_merge")

        print("Urutan test disesuaikan dengan template.")
    else:
        print(
            "Peringatan: ID bersama tidak ditemukan. "
            "Urutan asli data test dipertahankan."
        )

    # 4. Menyiapkan fitur sesuai metadata training.
    missing_features = [
        col for col in feature_columns
        if col not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Fitur test tidak lengkap: {missing_features}"
        )

    X_test = engineer_features(
        test_df[feature_columns],
        expected_columns=feature_columns
    )

    # 5. Melakukan prediksi.
    predictions = final_model.predict(X_test)

    if len(predictions) != len(sample_submission):
        raise ValueError("Jumlah prediksi tidak sesuai template.")

    if pd.isna(predictions).any():
        raise ValueError("Prediksi mengandung missing values.")

    # 6. Membuat submission sesuai format resmi.
    submission = sample_submission.copy()

    submission[target_column] = convert_predictions(
        predictions,
        sample_submission[target_column]
    ).to_numpy()

    submission = submission[sample_submission.columns]

    if submission.isna().any().any():
        raise ValueError("Submission mengandung missing values.")

    # 7. Menyimpan hasil submission.
    submission_dir.mkdir(parents=True, exist_ok=True)

    submission_path = submission_dir / "submission.csv"

    submission.to_csv(
        submission_path,
        index=False
    )

    # 8. Membaca kembali file untuk pemeriksaan.
    check = pd.read_csv(submission_path)

    if len(check) != len(test_df):
        raise ValueError(
            "Jumlah baris file submission tidak sesuai."
        )

    if check.columns.tolist() != sample_submission.columns.tolist():
        raise ValueError(
            "Kolom file submission tidak sesuai template."
        )

    if check.isna().any().any():
        raise ValueError(
            "File submission mengandung missing values."
        )

    print("\nSubmission berhasil dibuat.")
    print("Lokasi:", submission_path)
    print("Jumlah baris:", len(check))
    print("Kolom:", check.columns.tolist())
    print("\nContoh submission:")
    print(check.head())

    print("\nDistribusi target:")
    print(check[target_column].value_counts().sort_index())


if __name__ == "__main__":
    main()