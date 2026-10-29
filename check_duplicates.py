
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

original_path = (
    BASE_DIR / "data" /
    "Raw Hematological Dataset for Anemia Analysis and Classification.xlsx"
)

clean_path = BASE_DIR / "data" / "anemia_clean.csv"

original = pd.read_excel(original_path)
clean = pd.read_csv(clean_path)

original.columns = original.columns.str.strip()
clean.columns = clean.columns.str.strip()

# جست‌وجوی ردیف‌های دارای مقادیر مشکوک
checks = {
    "MCV": 120,
    "MCH": 40,
    "MCHC": 40,
    "RBC": 8
}

for column, limit in checks.items():
    print(f"\n{'=' * 40}")
    print("Checking:", column)

    original_bad = original[
        original[column] > limit
    ]

    clean_bad = clean[
        clean[column] > limit
    ]

    print("\nOriginal Excel:")
    print(
        original_bad[
            ["Gender", "Age", "HGB(Hemoglobin)",
             "RBC", "PCV/HCT", "MCV", "MCH",
             "MCHC", "Decision_Class"]

        ].to_string(index=False)
        if not original_bad.empty
        else "No rows found"
    )

    print("\nCleaned CSV:")
    print(
        clean_bad[
            ["Gender", "Age", "HGB(Hemoglobin)",
             "RBC", "PCV/HCT", "MCV", "MCH",
             "MCHC", "Decision_Class"]
        ].to_string(index=False)
        if not clean_bad.empty
        else "No rows found"
    )
