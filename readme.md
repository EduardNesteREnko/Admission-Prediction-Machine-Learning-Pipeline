# Admission Prediction Machine Learning Pipeline

This project implements a complete end-to-end Machine Learning pipeline to predict the chance of university admission (`Chance of Admit`) based on applicant academic metrics (GRE, TOEFL, CGPA, University Rating, SOP, LOR, and Research experience). 

The pipeline includes exploratory data analysis, feature engineering, dataset splitting, feature scaling, training a baseline Linear Regression model, and optimizing a Ridge Regression model using Grid Search cross-validation.

---

## 🚀 Key Features & Pipeline Steps

1. **Data Ingestion & Hygiene**:
   * Loads the dataset (`Admission_Predict.csv`).
   * Cleans column names by stripping accidental whitespaces.
   * Includes structural hooks for missing value checks and duplicate verification.

2. **Feature Engineering**:
   * **`Total_Score`**: Combines academic and linguistic preparation metrics (`GRE Score + TOEFL Score`).
   * **`Academic_Power`**: Evaluates the weight of acquired scores considering the prestige of the university (`CGPA * University Rating`).

3. **Data Splitting**:
   * Splits the data into **Train (70%)**, **Validation (15%)**, and **Test (15%)** subsets using robust random seeding (`random_state=42`).

4. **Feature Scaling**:
   * Applies `StandardScaler` to normalize features across all subsets, preventing scale dominance and improving regularization performance.

5. **Model Training & Hyperparameter Tuning**:
   * **Baseline Model**: Trains a standard `LinearRegression` model and evaluates performance on the validation set using the $R^2$ score.
   * **Advanced Model**: Trains a `Ridge` regression model and utilizes `GridSearchCV` (5-fold cross-validation) to find the optimal regularization parameter (`alpha` tested across `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`).

6. **Evaluation Metrics**:
   * Evaluates final model performance on the unseen test dataset using **$R^2$ (Coefficient of Determination)** and **MAE (Mean Absolute Error)**.

---

## 📦 Requirements

Ensure you have Python installed along with the required machine learning and data manipulation libraries:

```bash
pip install pandas numpy scikit-learn
```

---

## 🛠️ Dataset

This script expects the dataset file **`Admission_Predict.csv`** to be located in the same directory as the script. (Commonly sourced from Kaggle or the UCI Machine Learning repository).

---

## ▶️ Usage

Clone the repository, place your CSV file in the root folder, and run the script:

```bash
python script.py
```