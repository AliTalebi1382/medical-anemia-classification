
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
    confusion_matrix,
    classification_report
)

# 1. مسیرهای پروژه
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "anemia_clean.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 2. خواندن داده‌ها
df = pd.read_csv(DATA_PATH)

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

# 3. تبدیل جنسیت به عدد
X["Gender"] = (
    X["Gender"]
    .astype(str)
    .str.strip()
    .map({"F": 0, "M": 1})
)

if X.isna().any().any() or y.isna().any():
    raise ValueError("Missing or invalid data found.")

if set(y.unique()) != {0, 1}:
    raise ValueError("Expected target classes 0 and 1.")

# 4. تقسیم ثابت داده‌ها برای مقایسه منصفانه
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. تعریف سه مدل
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

# 6. اعتبارسنجی متقاطع روی داده آموزشی
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

summary_results = []
class_results = []

for name, model in models.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # ارزیابی متقاطع فقط روی داده آموزشی
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # آموزش و ارزیابی روی مجموعه آزمون ثابت
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

    # ماتریس اغتشاش با ترتیب کلاس‌های 0 و 1
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall   : {recall:.4f}")
    print(f"Test F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print([[tn, fp], [fn, tp]])

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Class 0", "Class 1"],
        zero_division=0
    ))

    result = {
        "Model": name,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1-score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    # میانگین و انحراف معیار هر معیار در 5-fold CV
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        result[f"CV {metric} mean"] = scores.mean()
        result[f"CV {metric} std"] = scores.std()

    summary_results.append(result)

    # ذخیره معیارهای جداگانه هر کلاس
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Class 0", "Class 1"],
        output_dict=True,
        zero_division=0
    )

    for class_name in ["Class 0", "Class 1"]:
        class_results.append({
            "Model": name,
            "Class": class_name,
            "Precision": report[class_name]["precision"],
            "Recall": report[class_name]["recall"],
            "F1-score": report[class_name]["f1-score"],
            "Support": report[class_name]["support"]
        })

# 7. ذخیره گزارش‌ها در فایل‌های جدید
summary_df = pd.DataFrame(summary_results)
summary_df = summary_df.sort_values(
    by="CV f1 mean",
    ascending=False
)

summary_path = RESULTS_DIR / "final_model_validation.csv"
summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

class_df = pd.DataFrame(class_results)
class_path = RESULTS_DIR / "final_classification_report.csv"
class_df.to_csv(
    class_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("FINAL VALIDATION SUMMARY")
print("=" * 60)
print(summary_df.round(4).to_string(index=False))

print("\nBest model by CV F1 mean:")
print(summary_df.iloc[0]["Model"])

print("\nFiles saved:")
print(summary_path)
print(class_path)
