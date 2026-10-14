# Structural Damage Classification

**Tugas 1 — Supervised Machine Learning 2026**

## 1. Project Overview

Project ini bertujuan mengklasifikasikan tingkat kerusakan bangunan pascagempa berdasarkan karakteristik struktural dan informasi bangunan menggunakan pendekatan *Supervised Machine Learning*.

Target klasifikasi yang digunakan adalah `damage_grade`, yang terdiri dari lima kelas tingkat kerusakan, yaitu Grade 1 hingga Grade 5.

## 2. Dataset

Dataset diperoleh melalui kompetisi Kaggle yang disediakan untuk tugas ini.

File dataset yang digunakan meliputi:

- `train.csv` — data training yang memuat fitur dan target.
- `test.csv` — data yang digunakan untuk menghasilkan prediksi.
- `sample_submission.csv` — template format file submission.

### Cara Mengunduh Dataset

1. Buka [halaman kompetisi Kaggle](https://www.kaggle.com/t/a9196b92680b4492aeb06aeaee94f654).
2. Klik **Join Competition** dan setujui peraturan kompetisi jika diminta.
3. Buka tab **Data**.
4. Klik **Download All** untuk mengunduh dataset.
5. Ekstrak file yang diunduh, kemudian letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` ke dalam folder `data/`.

File dataset berformat CSV tidak disimpan di GitHub karena dikecualikan melalui `.gitignore`.

## 3. Metodologi

Workflow proyek terdiri dari beberapa tahap berikut.

1. **Exploratory Data Analysis (EDA)**  
   Memeriksa karakteristik dataset, tipe data, missing values, duplikasi, statistik deskriptif, distribusi fitur, dan distribusi target.

2. **Preprocessing**  
   Mengidentifikasi fitur numerik dan kategorik serta menyiapkan penanganan missing values, scaling, dan encoding sesuai tipe fitur.

3. **Feature Engineering**  
   Menyiapkan fitur agar konsisten antara data training dan data test serta mempertahankan urutan kolom yang dibutuhkan model.

4. **Modeling**  
   Melatih dan membandingkan beberapa algoritma klasifikasi serta melakukan pengujian parameter untuk memperoleh model dengan performa terbaik.

5. **Evaluation**  
   Mengevaluasi model menggunakan Macro F1-Score, accuracy, classification report, dan confusion matrix.

6. **Prediction**  
   Menggunakan model final untuk menghasilkan prediksi pada data test dan menyimpannya dalam format submission yang sesuai dengan template.

Preprocessing dan model disusun menggunakan `Pipeline` dan `ColumnTransformer` dari Scikit-learn agar proses pengolahan fitur dan prediksi tetap konsisten.

## 4. Model dan Hasil Evaluasi

Algoritma klasifikasi yang dievaluasi dalam proyek ini meliputi:

- Logistic Regression
- Decision Tree
- Linear SVM
- Random Forest
- LightGBM

LightGBM memperoleh performa validasi terbaik di antara model yang telah diuji.

| Metrik Evaluasi | Skor Validasi |
|---|---:|
| Macro F1-Score | 0.538860 |
| Accuracy | 0.567207 |

Macro F1-Score digunakan sebagai metrik utama karena memperhitungkan performa prediksi pada setiap kelas.

### Interpretasi Hasil

Hasil evaluasi menunjukkan bahwa LightGBM menjadi model terbaik di antara algoritma yang telah diuji pada eksperimen proyek ini.

Berdasarkan classification report, Grade 2 dan Grade 3 merupakan kelas yang lebih sulit dikenali dibandingkan kelas lainnya. Recall kedua kelas tersebut masing-masing sebesar 0.43 dan 0.40. Hal ini menunjukkan bahwa model masih mengalami kesulitan dalam mengidentifikasi sebagian bangunan pada tingkat kerusakan tersebut.

Temuan ini menjadi dasar untuk pengembangan selanjutnya, khususnya melalui tuning parameter dan evaluasi kesalahan prediksi agar kemampuan model dalam membedakan tingkat kerusakan dapat ditingkatkan.

## 5. Struktur Repository

```text
structural-damage-classification/
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_submission.ipynb
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── generate_report.py
├── models/
│   ├── final_damage_model.joblib
│   └── model_metadata.json
├── submissions/
│   └── submission.csv
├── reports/
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   └── figures/
│       ├── confusion_matrix.png
│       └── feature_importance.png
├── README.md
├── requirements.txt
└── .gitignore
```

### Keterangan Struktur

- `data/` — menyimpan dataset lokal yang tidak diunggah ke GitHub.
- `notebooks/` — berisi notebook EDA, eksperimen modeling, dan pembuatan submission.
- `scripts/` — berisi fungsi preprocessing, feature engineering, training, prediksi, dan pembuatan laporan.
- `models/` — menyimpan model final dan metadata model.
- `submissions/` — menyimpan file hasil prediksi.
- `reports/` — menyimpan hasil perbandingan model, feature importance, dan visualisasi evaluasi.
- `requirements.txt` — mencatat dependency Python yang digunakan.
- `.gitignore` — menentukan file dan folder yang tidak disertakan dalam Git.

## 6. Cara Menjalankan Repository

### Persiapan Environment

Jalankan perintah berikut dari direktori utama project menggunakan terminal.

Buat virtual environment:

```powershell
python -m venv project_sml
```

Aktifkan environment pada Windows PowerShell:

```powershell
.\project_sml\Scripts\Activate.ps1
```

Instal dependency:

```powershell
pip install -r requirements.txt
```

Pastikan dataset sudah tersedia di folder `data/` sebelum menjalankan script.

### Menjalankan Script

Jalankan perintah berikut secara berurutan dari direktori utama project.

Training model:

```powershell
python -m scripts.train
```

Menghasilkan prediksi dan file submission:

```powershell
python -m scripts.predict
```

Menghasilkan laporan feature importance:

```powershell
python -m scripts.generate_report
```

### Menjalankan Notebook

Notebook dapat dijalankan secara berurutan sesuai workflow berikut:

1. `01_eda.ipynb` — eksplorasi dan pemahaman dataset.
2. `02_modeling.ipynb` — preprocessing, eksperimen model, dan evaluasi.
3. `03_submission.ipynb` — pembuatan dan pemeriksaan file submission.

## 7. Tools and Technologies

- **Python** — bahasa pemrograman utama.
- **Pandas dan NumPy** — manipulasi dan pengolahan data.
- **Scikit-learn** — preprocessing, Pipeline, ColumnTransformer, dan evaluasi model.
- **LightGBM** — algoritma klasifikasi final.
- **Matplotlib dan Seaborn** — visualisasi hasil analisis dan evaluasi.
- **Jupyter Notebook** — eksplorasi dataset dan eksperimen modeling.