
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Read the dataset
DATA_PATH = Path(__file__).parent / "data" / "anemia data from Kaggle.csv"
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# 2. Remove duplicate rows
df = df.drop_duplicates().copy()
print("Clean dataset shape:", df.shape)

# 3. Select input features and target
features = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]

X = df[features]
y = df["Result"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Create the AI model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

# 6. Train the model
model.fit(X_train, y_train)
print("\nModel training completed!")

# 7. Evaluate the model
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save the trained model
model_path = Path(__file__).parent / "anemia_model.joblib"
joblib.dump(model, model_path)
print("\nModel saved to:", model_path)

# 9. Ask the user for new data
print("\n--- Anemia Prediction ---")
print("Enter the person's blood test information.")

try:
    gender = int(input("Gender (0 or 1): "))
    hemoglobin = float(input("Hemoglobin: "))
    mch = float(input("MCH: "))
    mchc = float(input("MCHC: "))
    mcv = float(input("MCV: "))

    if gender not in [0, 1]:
        raise ValueError("Gender must be 0 or 1.")

    new_data = pd.DataFrame([{
        "Gender": gender,
        "Hemoglobin": hemoglobin,
        "MCH": mch,
        "MCHC": mchc,
        "MCV": mcv
    }])

    prediction = model.predict(new_data)[0]

    print("\nPredicted dataset class:", prediction)

    # Verify this label mapping against the original Kaggle dataset.
    if prediction == 1:
        print("Model result: Anemia class")
    else:
        print("Model result: Non-anemia class")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(new_data)[0]
        print("\nEstimated model probabilities:")
        for label, probability in zip(model.classes_, probabilities):
            print(f"Class {label}: {probability:.2%}")

except ValueError as error:
    print("Invalid input:", error)
