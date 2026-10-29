
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load the dataset
data_path = (
    Path(__file__).parent
    / "data"
    / "anemia data from Kaggle.csv"
)

df = pd.read_csv(data_path)

print("=" * 50)
print("1. DATASET INFORMATION")
print("=" * 50)

print("Original rows:", len(df))
print("Columns:", list(df.columns))
print("Missing values:\n", df.isnull().sum())

# 2. Check duplicate rows
print("\n" + "=" * 50)
print("2. DUPLICATE ANALYSIS")
print("=" * 50)

print("Exact duplicate rows:", df.duplicated().sum())

clean_df = df.drop_duplicates().copy()

print("Rows after removing exact duplicates:", len(clean_df))

# Check whether identical input measurements have conflicting labels
features = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
target = "Result"

label_counts = clean_df.groupby(features)[target].nunique()
conflicts = label_counts[label_counts > 1]

print(
    "Input combinations with conflicting labels:",
    len(conflicts)
)

# 3. Check class balance
print("\n" + "=" * 50)
print("3. CLASS DISTRIBUTION")
print("=" * 50)

print(clean_df[target].value_counts().sort_index())
print(
    "Class proportions:\n",
    clean_df[target].value_counts(normalize=True).sort_index().round(3)
)

# 4. Prepare the data
X = clean_df[features]
y = clean_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

# 5. Train and evaluate on a held-out test set
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n" + "=" * 50)
print("4. TEST SET EVALUATION")
print("=" * 50)

print(f"Test samples: {len(y_test)}")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(
    f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}"
)
print(
    f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}"
)
print(
    f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}"
)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# 6. Cross-validation on the cleaned dataset
print("\n" + "=" * 50)
print("5. FIVE-FOLD CROSS-VALIDATION")
print("=" * 50)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"]
)

for metric in ["accuracy", "precision", "recall", "f1"]:
    values = scores[f"test_{metric}"]
    print(
        f"{metric}: {values.mean():.4f} "
        f"+/- {values.std():.4f}"
    )

print("\nValidation completed.")
