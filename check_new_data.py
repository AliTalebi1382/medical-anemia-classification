
from pathlib import Path
import pandas as pd

file_path = (
    Path(__file__).parent
    / "data"
    / "Raw Hematological Dataset for Anemia Analysis and Classification.xlsx"
)

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# اصلاح فاصله‌های اضافی در مقادیر متنی
df["Gender"] = df["Gender"].astype("string").str.strip()

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

print("=" * 55)
print("1. DUPLICATE ROWS")
print("=" * 55)

duplicates = df[df.duplicated(keep=False)]
print("Rows belonging to duplicate groups:", len(duplicates))
print(duplicates.sort_values(features).to_string(index=False))

print("\n" + "=" * 55)
print("2. MISSING PCV/HCT")
print("=" * 55)

print(df[df["PCV/HCT"].isna()].to_string(index=False))

print("\n" + "=" * 55)
print("3. CONFLICTING LABELS")
print("=" * 55)

conflicts = (
    df.groupby(features, dropna=False)["Decision_Class"]
    .nunique()
)

conflict_keys = conflicts[conflicts > 1].index

if len(conflict_keys) == 0:
    print("No conflicting labels found.")
else:
    for key in conflict_keys:
        mask = pd.Series(True, index=df.index)

        for column, value in zip(features, key):
            if pd.isna(value):
                mask &= df[column].isna()
            else:
                mask &= df[column].eq(value)

        print(df.loc[mask].to_string(index=False))
        print("-" * 40)

print("\nInspection completed.")
