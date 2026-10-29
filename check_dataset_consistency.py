
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "anemia_clean.csv"
OUTPUT_PATH = BASE_DIR / "results" / "data_consistency_report.csv"

df = pd.read_csv(DATA_PATH)

# Calculate expected values from related blood measurements
df["MCV_expected"] = df["PCV/HCT"] * 10 / df["RBC"]
df["MCH_expected"] = df["HGB(Hemoglobin)"] * 10 / df["RBC"]
df["MCHC_expected"] = (
    df["HGB(Hemoglobin)"] * 100 / df["PCV/HCT"]
)

# Calculate absolute differences
df["MCV_difference"] = (df["MCV"] - df["MCV_expected"]).abs()
df["MCH_difference"] = (df["MCH"] - df["MCH_expected"]).abs()
df["MCHC_difference"] = (df["MCHC"] - df["MCHC_expected"]).abs()

# Screening thresholds, not medical diagnostic limits
thresholds = {
    "MCV_difference": 5,
    "MCH_difference": 3,
    "MCHC_difference": 3,
}

print("Total records:", len(df))

for column, threshold in thresholds.items():
    count = int((df[column] > threshold).sum())
    print(
        f"{column} > {threshold}: "
        f"{count} records ({count / len(df) * 100:.2f}%)"
    )

# Save all records with calculated differences
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\nReport saved to:", OUTPUT_PATH)
