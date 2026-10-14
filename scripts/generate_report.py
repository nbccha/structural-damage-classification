"""Membuat laporan feature importance dari model LightGBM final."""

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def main():
    """Menghasilkan grafik dan tabel feature importance."""

    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root / "models" / "final_damage_model.joblib"
    )

    report_dir = project_root / "reports"
    figure_dir = report_dir / "figures"

    figure_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model final tidak ditemukan: {model_path}. "
            "Jalankan scripts/train.py terlebih dahulu."
        )

    # Memuat model final.
    pipeline = joblib.load(model_path)

    if "prep" not in pipeline.named_steps:
        raise ValueError(
            "Tahap preprocessing 'prep' tidak ditemukan."
        )

    if "algo" not in pipeline.named_steps:
        raise ValueError(
            "Model 'algo' tidak ditemukan dalam pipeline."
        )

    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["algo"]

    # Mendapatkan nama fitur setelah preprocessing.
    feature_names = preprocessor.get_feature_names_out()

    # Menggunakan gain untuk mengukur kontribusi fitur
    # terhadap perbaikan objective selama pembentukan pohon.
    importance_values = model.booster_.feature_importance(
        importance_type="gain"
    )

    if len(feature_names) != len(importance_values):
        raise ValueError(
            "Jumlah nama fitur tidak sesuai dengan jumlah importance."
        )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    # Mengurutkan fitur berdasarkan importance.
    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    # Menyimpan tabel lengkap.
    csv_path = report_dir / "feature_importance.csv"

    importance_df.to_csv(
        csv_path,
        index=False
    )

    # Mengambil 20 fitur teratas untuk visualisasi.
    top_features = importance_df.head(20).copy()

    if top_features["Importance"].sum() == 0:
        raise ValueError(
            "Semua nilai feature importance bernilai nol."
        )

    # Membuat grafik horizontal.
    top_features = top_features.sort_values(
        "Importance",
        ascending=True
    )

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title("Top 20 Feature Importance - LightGBM")
    plt.xlabel("Importance (Gain)")
    plt.ylabel("Feature")
    plt.tight_layout()

    figure_path = figure_dir / "feature_importance.png"

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Laporan berhasil dibuat.")
    print("Grafik:", figure_path)
    print("Tabel :", csv_path)

    print("\n10 fitur teratas:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()