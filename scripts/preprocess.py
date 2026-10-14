"""Fungsi preprocessing untuk klasifikasi kerusakan bangunan."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def infer_target_column(train_data, test_data, sample_data=None):
    """
    Menentukan kolom target berdasarkan struktur data train dan test.
    """

    if sample_data is not None:
        sample_candidates = [
            col for col in sample_data.columns
            if col in train_data.columns
            and col not in test_data.columns
        ]

        if len(sample_candidates) == 1:
            return sample_candidates[0]

    target_candidates = [
        col for col in train_data.columns
        if col not in test_data.columns
    ]

    if len(target_candidates) != 1:
        raise ValueError(
            "Target tidak dapat ditentukan secara otomatis. "
            f"Kandidat target: {target_candidates}"
        )

    return target_candidates[0]


def detect_id_columns(data):
    """
    Mendeteksi kolom yang kemungkinan berfungsi sebagai identifier.
    """

    id_columns = []
    n_rows = max(len(data), 1)

    for col in data.columns:
        col_name = str(col).lower().strip().replace(" ", "_")

        unique_ratio = (
            data[col].nunique(dropna=False) / n_rows
        )

        n_unique = data[col].nunique(dropna=False)

        named_id = (
            col_name in {"id", "index"}
            or col_name.endswith("_id")
            or "identifier" in col_name
            or col_name.endswith("_uuid")
        )

        minimum_unique = min(
            100,
            max(10, int(n_rows * 0.5))
        )

        high_unique = (
            unique_ratio >= 0.98
            and n_unique >= minimum_unique
        )

        if (
            (named_id and unique_ratio >= 0.5)
            or high_unique
        ):
            id_columns.append(col)

    return id_columns


def identify_feature_types(data):
    """
    Mengidentifikasi fitur numerik dan kategorik.

    Fitur numerik berkardinalitas rendah atau fitur dengan nama
    yang mengindikasikan kode kategori dapat diperlakukan
    sebagai fitur kategorik.
    """

    categorical_columns = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_keywords = (
        "type",
        "status",
        "condition",
        "category",
        "class",
        "grade",
        "level",
        "material",
        "position",
        "configuration",
        "ownership",
        "surface",
        "foundation",
        "roof",
        "district",
        "ward",
        "municip",
        "code",
        "_id"
    )

    for col in numerical_columns.copy():
        col_name = str(col).lower()

        is_integer = pd.api.types.is_integer_dtype(
            data[col]
        )

        is_categorical_code = (
            any(
                keyword in col_name
                for keyword in categorical_keywords
            )
            or (
                is_integer
                and data[col].nunique(dropna=True) <= 12
            )
        )

        if is_categorical_code:
            numerical_columns.remove(col)

            if col not in categorical_columns:
                categorical_columns.append(col)

    return numerical_columns, categorical_columns


def build_preprocessor(numerical_columns, categorical_columns):
    """
    Membuat ColumnTransformer untuk fitur numerik dan kategorik.

    Fitur numerik menggunakan median imputation dan MinMaxScaler.
    Fitur kategorik menggunakan imputasi modus dan OneHotEncoder.
    """

    if set(numerical_columns) & set(categorical_columns):
        raise ValueError(
            "Kolom numerik dan kategorik tidak boleh tumpang tindih."
        )

    transformers = []

    if numerical_columns:
        numerical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                MinMaxScaler()
            )
        ])

        transformers.append((
            "numeric",
            numerical_pipeline,
            numerical_columns
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns
        ))

    if not transformers:
        raise ValueError(
            "Tidak ada fitur numerik atau kategorik untuk diproses."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )