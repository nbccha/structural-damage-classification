import numpy as np
import pandas as pd


def engineer_features(data, expected_columns=None):
    """Menyiapkan fitur dan menjaga konsistensi urutan kolom."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame.")

    features = data.copy()

    if expected_columns is not None:
        missing_columns = [
            col for col in expected_columns
            if col not in features.columns
        ]

        if missing_columns:
            raise ValueError(
                "Kolom fitur tidak ditemukan: "
                f"{missing_columns}"
            )

        features = features[expected_columns].copy()

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return features