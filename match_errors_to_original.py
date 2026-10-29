
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = (
    BASE_DIR
    / "data"
    / "Raw Hematological Dataset for Anemia Analysis and Classification.xlsx"
)

CLEAN_PATH = BASE_DIR / "data" / "anemia_clean.csv"
ERROR_PATH = BASE_DIR / "results" / "prediction_errors.csv"

original = pd.read_excel(ORIGINAL_PATH)
clean = pd.read_csv(CLEAN_PATH)
errors = pd.read_csv(ERROR_PATH)

# Normalize column names and gender values
for df in (original, clean, errors):
    df.columns = df.columns.str.strip()

original["Gender"] = original["Gender"].astype(str).str.strip().str.upper()
clean["Gender"] = clean["Gender"].astype(str).str.strip().str.upper()

# Features used to identify each sample
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

numeric_features = [col for col in features if col != "Gender"]

for df in (original, clean, errors):
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("Matching incorrect predictions to original Excel data")

for _, error in errors.iterrows():
    mask = pd.Series(True, index=original.index)

    for col in features:
        if col == "Gender":
            mask &= original[col] == str(
                int(error[col])
            ).strip() if False else original[col].isin(["F", "M"])
            expected_gender = "M" if int(error[col]) == 1 else "F"
            mask &= original[col] == expected_gender
        else:
            mask &= original[col].eq(error[col])

    matches = original.loc[mask]

    print("\n" + "=" * 55)
    print(
        "Cleaned sample:",
        "Original_Row_Index =", int(error["Original_Row_Index"]),
    )
    print(
        "Actual class:",
        int(error["Actual_Class"]),
        "| Predicted class:",
        int(error["Predicted_Class"]),
    )

    if matches.empty:
        print("No exact matching row found in original Excel.")
    else:
        print("Matching rows in original Excel:")
        print(matches.to_string(index=False))
        print("Number of matches:", len(matches))

