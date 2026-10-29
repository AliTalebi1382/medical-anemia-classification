
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. مسیرهای پروژه
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "anemia_clean.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 2. خواندن دیتاست پاک‌سازی‌شده
df = pd.read_csv(DATA_PATH)

# 3. مشخص کردن ورودی‌ها و خروجی
features = [
    "Gender",
    "Age",
    "HGB(Hemoglobin)",
    "RBC",
    "PCV/HCT",
    "MCV",
    "MCH",
    "MCHC"
]

target = "Decision_Class"

X = df[features].copy()
y = df[target].astype(int)

# تبدیل جنسیت متنی به مقدار عددی
X["Gender"] = X["Gender"].map({"F": 0, "M": 1})

if X.isna().any().any():
    raise ValueError(
        "Missing or unrecognized values found in input features."
    )

# 4. تقسیم یکسان داده برای هر سه مدل
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. تعریف مدل‌ها
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
}

# 6. آموزش، ارزیابی و ذخیره
results = []
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Class 0", "Class 1"],
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # ارزیابی متقاطع فقط روی بخش آموزشی
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1"
        },
        n_jobs=-1
    )

    cv_f1_mean = cv_scores["test_f1"].mean()
    cv_f1_std = cv_scores["test_f1"].std()

    print(
        f"\n5-fold CV F1: "
        f"{cv_f1_mean:.4f} +/- {cv_f1_std:.4f}"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "CV F1 mean": cv_f1_mean,
        "CV F1 std": cv_f1_std
    })

    # ذخیره مدل آموزش‌دیده
    model_path = MODEL_DIR / (
        name.lower().replace(" ", "_") + ".joblib"
    )
    joblib.dump(model, model_path)
    print("Saved model:", model_path)

# 7. ذخیره جدول مقایسه
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="CV F1 mean",
    ascending=False
)


comparison_path = RESULTS_DIR / "model_comparison.csv"
results_df.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(results_df.round(4).to_string(index=False))
print("\nComparison saved to:", comparison_path)

print(
    "\nBest model by cross-validation F1:",
    results_df.iloc[0]["Model"]
)

print("Training and evaluation completed.")
