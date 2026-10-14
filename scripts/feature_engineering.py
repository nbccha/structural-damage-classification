"""Fungsi penyiapan fitur untuk model klasifikasi kerusakan bangunan."""

import numpy as np
import pandas as pd


def engineer_features(data, expected_columns=None):
    """
    Menyiapkan fitur agar konsisten antara data training dan test.

    Parameters
    ----------
    data : pandas.DataFrame
        Data fitur yang akan diproses.
    expected_columns : list, optional
        Urutan kolom yang digunakan saat training.

    Returns
    -------
    pandas.DataFrame
        Data fitur yang telah disiapkan.
    """

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

    # Menghindari nilai tak hingga saat preprocessing.
    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return features