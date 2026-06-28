from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = [
    "Age",
    "Resting BP",
    "Serum Cholestrol(mg/dl)",
    "Max Heart Rate",
    "Oldpeak",
    "No. of Major Vessels"
]

BOOLEAN_FEATURES = [
    "Fasting Blood Sugar (> 120 mg/dl)",
    "Angina (Exercise Induced)"
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Chest Pain",
    "Resting ECG Result",
    "Slope",
    "Thal"
]

def build_preprocessor() -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=10)),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor