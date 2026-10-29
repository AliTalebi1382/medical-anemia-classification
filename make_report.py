
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

# File paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "anemia data from Kaggle.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load and clean data
df = pd.read_csv(DATA_PATH).drop_duplicates()

features = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
X = df[features]
y = df["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train the model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Chart 1: Class distribution
class_counts = y.value_counts().sort_index()

plt.figure(figsize=(7, 5))
class_counts.plot(kind="bar")
plt.title("Anemia Dataset: Class Distribution")
plt.xlabel("Class (0 = Non-anemia, 1 = Anemia)")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "class_distribution.png",
    dpi=200
)
plt.close()

# Chart 2: Confusion matrix
display = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Non-anemia", "Anemia"],
    cmap="Blues",
    values_format="d"
)

display.ax_.set_title("Logistic Regression - Confusion Matrix")
display.figure_.tight_layout()
display.figure_.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=200
)
plt.close(display.figure_)

print("Report charts created successfully!")
print("Saved to:", RESULTS_DIR)
print("Files:")
print("- class_distribution.png")
print("- confusion_matrix.png")
