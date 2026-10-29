
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "anemia_clean.csv"

df = pd.read_csv(DATA_PATH)

print("Class counts:")
print(df["Decision_Class"].value_counts().sort_index())

print("\nMean blood measurements by class:")
columns = [
    "HGB(Hemoglobin)",
    "RBC",
    "PCV/HCT",
    "MCV",
    "MCH",
    "MCHC"
]

print(
    df.groupby("Decision_Class")[columns]
    .mean()
    .round(2)
    .to_string()
)

print("\nMedian hemoglobin by class:")
print(
    df.groupby("Decision_Class")["HGB(Hemoglobin)"]
    .median()
)

print("\nHemoglobin ranges by class:")
print(
    df.groupby("Decision_Class")["HGB(Hemoglobin)"]
    .agg(["min", "max", "mean", "median"])
    .round(2)
)
