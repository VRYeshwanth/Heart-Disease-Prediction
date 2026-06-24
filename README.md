# Heart Disease Prediction using Machine Learning

A machine learning project that predicts the presence of heart disease using clinical and diagnostic attributes from the Cleveland Heart Disease dataset. The project includes Exploratory Data Analysis (EDA), data preprocessing, model comparison, hyperparameter tuning, and final model selection.

---

## 🎯 Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection can help healthcare professionals provide timely treatment and improve patient outcomes.

This project aims to build and evaluate multiple machine learning models for heart disease prediction and identify the best-performing model based on classification metrics, with particular emphasis on **Recall**, as minimizing false negatives is important in healthcare applications.

---

## 📊 Dataset

### Features

| Feature                          | Description                                    |
| -------------------------------- | ---------------------------------------------- |
| Age                              | Age of the patient                             |
| Sex                              | Gender of the patient                          |
| Chest Pain                       | Type of chest pain experienced                 |
| Resting BP                       | Resting blood pressure                         |
| Serum Cholestrol (mg/dl)         | Serum cholesterol level                        |
| Fasting Blood Sugar (>120 mg/dl) | Fasting blood sugar indicator                  |
| Resting ECG Result               | Resting electrocardiographic results           |
| Max Heart Rate                   | Maximum heart rate achieved                    |
| Angina (Exercise Induced)        | Exercise-induced angina                        |
| Oldpeak                          | ST depression induced by exercise              |
| Slope                            | Slope of the peak exercise ST segment          |
| No. of Major Vessels             | Number of major vessels colored by fluoroscopy |
| Thal                             | Thalassemia status                             |
| Target                           | Presence or absence of heart disease           |

### Dataset Shape

- Rows: 304
- Columns: 14

---

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Analyzed feature distributions using histograms and boxplots.
- Examined categorical feature distributions using count plots.
- Identified skewness and outliers in numerical variables.
- Investigated relationships between features and the target variable.
- Generated a correlation heatmap to understand feature interactions.

### 2. Data Preprocessing

- Handled categorical variables using One-Hot Encoding.
- Applied feature scaling where required.
- Split data into training and testing sets.
- Built preprocessing pipelines for reproducibility.

### 3. Model Training

The following classification models were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost

### 4. Hyperparameter Tuning

GridSearchCV was used to optimize model performance and identify the best hyperparameters.

### 5. Model Evaluation

Models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score

---

## 🏆 Results

### Baseline Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 82.87%   |
| KNN                 | 83.85%   |
| Naive Bayes         | 80.89%   |
| SVM                 | 80.90%   |
| Decision Tree       | 74.32%   |
| Random Forest       | 80.91%   |
| XGBoost             | 79.93%   |

### Tuned Model Performance

| Model               | Accuracy | Recall |
| ------------------- | -------- | ------ |
| Logistic Regression | 81.97%   | 86.21% |
| KNN                 | 86.89%   | 89.66% |
| Naive Bayes         | 85.25%   | 86.21% |
| SVM                 | 86.89%   | 86.21% |
| Decision Tree       | 70.49%   | 72.41% |
| Random Forest       | 83.61%   | 82.76% |
| XGBoost             | 81.97%   | 82.76% |

---

## ✅ Final Model Selection

### Support Vector Machine (SVM)

The SVM model was selected as the final model because:

- Achieved one of the highest accuracies (**86.89%**).
- Maintained strong recall (**86.21%**), which is important for identifying heart disease patients.
- Provides better generalization than instance-based methods such as KNN.
- Less sensitive to the curse of dimensionality compared to KNN.

---

## 📂 Repository Structure

```text
Heart-Disease-Prediction/
├── dataset/
│   └── heart_disease.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_comparison.ipynb
├── README.md
└── requirements.txt
```

---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/VRYeshwanth/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

2. Create Virtual Environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the notebooks in the following order:

1. `01_eda.ipynb`
2. `02_preprocessing.ipynb`
3. `03_model_comparison.ipynb`

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Jupyter Notebook

---

## 🎓 Key Learnings

- Performing Exploratory Data Analysis on medical datasets.
- Building preprocessing pipelines for machine learning workflows.
- Comparing multiple classification algorithms.
- Applying hyperparameter tuning using GridSearchCV.
- Evaluating models using healthcare-relevant metrics such as Recall.
- Selecting an appropriate model based on both performance and generalization.

---
