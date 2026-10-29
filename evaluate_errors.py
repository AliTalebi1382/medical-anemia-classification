
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# مسیرهای پروژه
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "anemia_clean.csv"
OUTPUT_PATH = BASE_DIR / "results" / "prediction_errors.csv"

# خواندن دیتاست
df = pd.read_csv(DATA_PATH)

# مشخص کردن ویژگی‌ها و ستون هدف
features = [
    "Gender",
    "Age",
    "HGB(Hemoglobin)",
    "RBC",
    "PCV/HCT",
    "MCV",
    "MCH",
    "MCHC",
]
target = "Decision_Class"

# تبدیل جنسیت به عدد
df["Gender"] = (
    df["Gender"].astype(str).str.strip().str.upper()
    .map({"F": 0, "M": 1})
)

# آماده‌سازی داده‌ها
X = df[features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[target], errors="coerce")

valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

# همان تقسیم‌بندی آموزش و آزمون قبلی
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# بارگذاری مدل ذخیره‌شده
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
model = joblib.load(MODEL_PATH)
# پیش‌بینی روی داده‌های آزمون
predictions = model.predict(X_test)

print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# استخراج نمونه‌های اشتباه
errors = X_test.copy()
errors["Actual_Class"] = y_test
errors["Predicted_Class"] = predictions
errors = errors.loc[errors["Actual_Class"] != errors["Predicted_Class"]].copy()

# افزودن احتمال‌های پیش‌بینی‌شده، در صورت پشتیبانی مدل
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_test)
    for index, class_label in enumerate(model.classes_):
        errors[f"Probability_Class_{class_label}"] = probabilities[
            errors.index.map(X_test.index.get_loc)
        ][:, index]

# ذخیره نتیجه
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
errors.to_csv(OUTPUT_PATH, index=True, index_label="Original_Row_Index",
              encoding="utf-8-sig")

print("\nNumber of incorrect predictions:", len(errors))
print("Error report saved to:", OUTPUT_PATH)
print("\nFirst incorrect predictions:")
print(errors.head(10).to_string())
