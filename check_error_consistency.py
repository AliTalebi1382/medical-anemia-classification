
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
ERROR_PATH = BASE_DIR / "results" / "prediction_errors.csv"

df = pd.read_csv(ERROR_PATH)

# Calculate expected blood-index values
df["MCV_expected"] = df["PCV/HCT"] * 10 / df["RBC"]
df["MCH_expected"] = df["HGB(Hemoglobin)"] * 10 / df["RBC"]
df["MCHC_expected"] = (
    df["HGB(Hemoglobin)"] * 100 / df["PCV/HCT"]
)

# Calculate differences between recorded and expected values
df["MCV_difference"] = (df["MCV"] - df["MCV_expected"]).abs()
df["MCH_difference"] = (df["MCH"] - df["MCH_expected"]).abs()
df["MCHC_difference"] = (df["MCHC"] - df["MCHC_expected"]).abs()

columns = [
    "Original_Row_Index",
    "Actual_Class",
    "Predicted_Class",
    "MCV",
    "MCV_expected",
    "MCV_difference",
    "MCH",
    "MCH_expected",
    "MCH_difference",
    "MCHC",
    "MCHC_expected",
    "MCHC_difference",
]

print(df[columns].round(2).to_string(index=False))
