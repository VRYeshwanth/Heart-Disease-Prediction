# 🩺 Heart Disease Classification using Machine Learning

## Project Overview

This project explores the application of various machine learning algorithms for predicting heart disease status. The workflow covers the complete machine learning pipeline, including exploratory data analysis (EDA), preprocessing, handling class imbalance, baseline model evaluation, hyperparameter tuning, and model comparison.

The primary objective was to identify the most effective classification model while investigating the impact of class imbalance and hyperparameter optimization on model performance.

---

## Dataset Information

🔗 Link to Dataset: [Click Here](https://www.kaggle.com/datasets/oktayrdeki/heart-disease/data)

### Dataset Characteristics

- Total Samples: 10,000
- Numerical Features: 9
- Categorical Features: 12
- Total Features After One-Hot Encoding: 35
- Target Variable: `Heart Disease Status`

### Target Distribution

| Class                | Percentage |
| -------------------- | ---------- |
| No Heart Disease (0) | 80%        |
| Heart Disease (1)    | 20%        |

The dataset exhibits significant class imbalance, making evaluation metrics such as Recall and F1 Score more important than Accuracy.

---

## Exploratory Data Analysis (EDA)

### Numerical Features

The following numerical features were analyzed:

- Age
- Blood Pressure
- Cholesterol Level
- BMI
- Sleep Hours
- Triglyceride Level
- Fasting Blood Sugar
- CRP Level
- Homocysteine Level

### EDA Findings

- Numerical features exhibited near-uniform distributions.
- Logarithmic and square-root transformations were explored.
- Outlier analysis using boxplots showed minimal extreme outliers.
- Numerical features displayed extremely weak correlations with one another.

### Correlation Analysis

The correlation matrix revealed:

- Very weak linear relationships between numerical variables.
- Most correlation coefficients were close to zero.
- No strong multicollinearity was observed.

### Categorical Features

The following categorical features were analyzed:

- Gender
- Exercise Habits
- Smoking
- Family Heart Disease
- Diabetes
- High Blood Pressure
- Low HDL Cholesterol
- High HDL Cholesterol
- Alcohol Consumption
- Stress Level
- Sugar Consumption

### Categorical Findings

- Binary categorical variables were approximately 50:50 distributed.
- Multi-class categorical variables were nearly evenly distributed across categories.
- Target variable remained the only significantly imbalanced feature.

---

## Data Preprocessing

### Steps Performed

1. Train-Test Split
2. Target Encoding using LabelEncoder
3. One-Hot Encoding for categorical variables
4. Standard Scaling for numerical variables
5. Column-wise transformations using ColumnTransformer
6. End-to-End preprocessing using Pipelines

### Preprocessing Pipeline

#### Numerical Features

- KNNImputer
- StandardScaler

#### Categorical Features

- SimpleImputer
- OneHotEncoder

---

## Handling Class Imbalance

To address the imbalance in the target variable, SMOTE (Synthetic Minority Oversampling Technique) was applied within the machine learning pipeline.

### Why SMOTE?

The minority class represented only 20% of the dataset. SMOTE was used to generate synthetic minority samples during training to improve minority-class detection.

---

## Models Evaluated

The following machine learning models were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost

---

## Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score

Since the dataset is imbalanced, F1 Score and Recall were prioritized over Accuracy.

---

## Baseline Results (SMOTE Applied)

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   0.5095 |    0.1899 | 0.4450 |   0.2663 |
| KNN                 |   0.4775 |    0.2081 | 0.5750 |   0.3056 |
| Naive Bayes         |   0.5955 |    0.1859 | 0.3025 |   0.2303 |
| SVM                 |   0.7075 |    0.1906 | 0.1425 |   0.1631 |
| Decision Tree       |   0.6595 |    0.1978 | 0.2300 |   0.2127 |
| Random Forest       |   0.7985 |    0.0000 | 0.0000 |   0.0000 |
| XGBoost             |   0.7705 |    0.1529 | 0.0325 |   0.0536 |

---

## Hyperparameter Tuning

GridSearchCV was used to optimize model hyperparameters.

### Cross Validation

- Grid Search Cross Validation
- Number of Folds: 10
- Scoring Metric: F1 Score

---

## Final Results After Hyperparameter Tuning

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   0.5095 |    0.1900 | 0.4450 |   0.2663 |
| KNN                 |   0.3890 |    0.2000 | 0.6850 |   0.3096 |
| Naive Bayes         |   0.5955 |    0.1859 | 0.3025 |   0.2303 |
| SVM                 |   0.3490 |    0.1965 | 0.7300 |   0.3097 |
| Decision Tree       |   0.6810 |    0.2069 | 0.2100 |   0.2084 |
| Random Forest       |   0.7985 |    0.2000 | 0.0025 |   0.0049 |
| XGBoost             |   0.7905 |    0.1481 | 0.0100 |   0.0187 |

---

## Key Findings

### Observations

- Hyperparameter tuning improved SVM significantly.
- KNN remained one of the strongest performers after tuning.
- Logistic Regression showed stable but limited performance.
- Random Forest and XGBoost consistently struggled to identify the minority class.
- SMOTE improved minority-class detection for KNN and SVM.

### Dataset Insights

Despite:

- Extensive preprocessing
- Class balancing using SMOTE
- Hyperparameter optimization
- Multiple classification algorithms

Performance improvements remained modest.

The EDA revealed:

- Nearly uniform feature distributions
- Weak relationships between numerical variables
- Limited separation between classes

These characteristics suggest that the dataset may contain limited predictive signal, restricting the achievable classification performance regardless of model complexity.

---

## Technologies Used

### Python Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn
- XGBoost

### Machine Learning Concepts

- Data Preprocessing
- Feature Engineering
- One-Hot Encoding
- Standardization
- Pipelines
- ColumnTransformer
- SMOTE
- Hyperparameter Tuning
- Cross Validation
- Classification Metrics

---

## Repository Structure

```text
Heart-Disease-Prediction/
├── dataset/
│   ├── cleaned_dataset.csv
│   └── heart_disease.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_comparison.ipynb
├── README.md
└── requirements.txt
```

---

## Conclusion

This project demonstrates a complete machine learning workflow for binary classification, including preprocessing, imbalance handling, model evaluation, and hyperparameter optimization.

While model performance remained modest, the project provided valuable insights into the limitations of the dataset and highlighted the importance of data quality and predictive signal in machine learning applications.
