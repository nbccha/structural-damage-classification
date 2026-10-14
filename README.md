# Structural Damage Classification

**Supervised Machine Learning Project**

## 1. Project Overview

This project aims to classify building damage levels based on structural characteristics using supervised machine learning. The project is based on the *Structural Damage Classification* Kaggle competition, which focuses on predicting building damage after an earthquake.

The objective is to develop and evaluate classification models that can predict the `damage_grade` of buildings.

## 2. Objectives

- Explore and understand the dataset.
- Prepare numerical and categorical features for modeling.
- Compare non-neural-network classification algorithms.
- Evaluate model performance using Macro F1-Score.
- Generate predictions in the required Kaggle submission format.

## 3. Dataset

The dataset is obtained from the Kaggle competition and includes:

| File | Description |
|---|---|
| `train.csv` | Training data containing building features and the target variable. |
| `test.csv` | Data used to generate predictions. |
| `sample_submission.csv` | Reference for the required submission format. |
| Metadata file | Additional information provided with the competition data. |

The dataset files are stored locally in the `data/` directory and are excluded from the GitHub repository.

## 4. Methodology

The project follows these steps:

1. **Exploratory Data Analysis (EDA):** Examine dataset structure, feature types, missing values, and target distribution.
2. **Preprocessing:** Handle missing values, scale numerical features, and encode categorical features.
3. **Model Training:** Train and compare classification algorithms using a validation dataset.
4. **Evaluation:** Assess model performance using Macro F1-Score, classification reports, and a confusion matrix.
5. **Prediction:** Generate predictions for the test dataset and prepare the submission file.

Preprocessing and model training are organized using `Pipeline` and `ColumnTransformer`.

## 5. Machine Learning Models

The following non-neural-network algorithms will be evaluated:

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline classification model. |
| Extra Trees Classifier | Tree-based ensemble model for learning more complex patterns. |

The final model will be selected based on validation results.

## 6. Evaluation Metric

The primary evaluation metric is **Macro F1-Score**, which calculates the F1-Score for each class and gives equal weight to all classes.

Additional evaluation uses a classification report and confusion matrix to examine model performance across the five `damage_grade` classes.

## 7. Project Structure

```text
structural-damage-classification/
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_submission.ipynb
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/
├── submissions/
├── reports/
│   └── figures/
├── README.md
├── requirements.txt
└── .gitignore
```

## 8. Tools and Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook
- Git and GitHub
- Kaggle

## 9. Installation and Usage

### Create a virtual environment

```powershell
python -m venv project_sml
.\project_sml\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Prepare the dataset

Download the competition files from Kaggle and place them in the `data/` directory, keeping their original filenames.

### Run the notebooks

Execute the notebooks in the following order:

1. `01_eda.ipynb`
2. `02_modeling.ipynb`
3. `03_submission.ipynb`

## 10. Results and Interpretation

Model performance, validation Macro F1-Scores, evaluation findings, and Kaggle leaderboard results will be documented after the experiments are completed.

The final interpretation will explain the selected model, its performance across classes, and its limitations.

## References

- [Structural Damage Classification — Kaggle Competition](https://www.kaggle.com/competitions/structural-damage-classification/overview)
- Tugas 1 — Supervised Machine Learning Assignment