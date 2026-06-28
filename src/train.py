from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from preprocessing import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "heart_disease.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "heart_disease_pipeline.pkl"


df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Target"])
y = df["Target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = build_preprocessor()

svm_model = SVC(
    C=0.01,
    kernel="linear",
    gamma="scale",
    class_weight="balanced",
    random_state=42,
    probability=True
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", svm_model)
])


print("Training model...")
pipeline.fit(X_train, y_train)
print("Training complete.")


MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"\nPipeline saved to:\n{MODEL_PATH}")