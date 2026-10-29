
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "anemia_clean.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["Decision_Class"].value_counts().sort_index())

print("\nMean blood measurements by class:")
columns = [
    "HGB(Hemoglobin)",
    "RBC",
    "PCV/HCT",
    "MCV",
    "MCH",
    "MCHC",
]

print(df.groupby("Decision_Class")[columns].mean().round(2))

print("\nSample rows from each class:")
for class_value in sorted(df["Decision_Class"].unique()):
    print(f"\nClass {class_value}:")
    print(
        df.loc[df["Decision_Class"] == class_value, columns]
        .head(5)
        .to_string(index=False)
    )
