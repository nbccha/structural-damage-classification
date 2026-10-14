# Structural Damage Classification

**Supervised Machine Learning Project**

## 1. Project Overview

This project applies supervised machine learning to classify building damage levels based on structural and building-related features. The target variable is `damage_grade`, representing five levels of building damage.

## 2. Dataset

The dataset is provided through the Kaggle competition used for this project. It consists of the following files:

- `train.csv` — training data containing target labels.
- `test.csv` — data used to generate predictions.
- `sample_submission.csv` — template for the prediction output.

### Download Instructions

1. Open the Kaggle competition page provided for this assignment.
2. Join the competition and accept the competition rules if required.
3. Navigate to the **Data** tab.
4. Click **Download All** to download the dataset files.
5. Extract the downloaded files and place `train.csv`, `test.csv`, and `sample_submission.csv` inside the project's `data/` directory.

Dataset CSV files are excluded from GitHub through `.gitignore`.

## 3. Methodology

The project workflow consists of three main stages:

1. **Exploratory Data Analysis (EDA):** Examine dataset characteristics, missing values, duplicate records, feature distributions, and target distribution.
2. **Modeling:** Perform data preprocessing, feature engineering, model comparison, and hyperparameter tuning.
3. **Evaluation and Prediction:** Evaluate model performance using Macro F1-Score, accuracy, classification reports, and confusion matrix analysis, then generate predictions for the test dataset.

The preprocessing and modeling workflow uses Scikit-learn `Pipeline` and `ColumnTransformer` to maintain consistency between training and prediction.

## 4. Models and Results

The evaluated classification algorithms include Logistic Regression, Decision Tree, Linear SVM, Random Forest, and LightGBM.

LightGBM achieved the best validation performance among the tested models.

| Evaluation Metric | Validation Score |
|---|---:|
| Macro F1-Score | 0.538860 |
| Accuracy | 0.567207 |

Macro F1-Score is the primary evaluation metric because it considers performance across all classes.

The classification report indicates that Grade 2 and Grade 3 are more difficult for the model to identify than the other damage classes, with recall scores of 0.43 and 0.40, respectively. These results indicate that distinguishing between certain damage levels remains a challenge and provides an opportunity for further model improvement.

## 5. Project Structure

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
│   ├── predict.py
│   └── generate_report.py
├── models/
│   ├── final_damage_model.joblib
│   └── model_metadata.json
├── submissions/
│   └── submission.csv
├── reports/
│   ├── feature_importance.csv
│   └── figures/
│       ├── confusion_matrix.png
│       └── feature_importance.png
├── README.md
├── requirements.txt
└── .gitignore
```

## 6. Tools and Technologies

- **Python** — primary programming language.
- **Pandas and NumPy** — data processing and manipulation.
- **Scikit-learn** — preprocessing, pipelines, model evaluation, and validation.
- **LightGBM** — final classification model.
- **Matplotlib and Seaborn** — data visualization and evaluation reports.
- **Jupyter Notebook** — exploratory data analysis and modeling experiments.