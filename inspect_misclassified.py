
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

# Normalize column names
original.columns = original.columns.str.strip()
clean.columns = clean.columns.str.strip()

# Read original row indices recorded in the error report
error_indices = errors["Original_Row_Index"].astype(int)

print("Incorrect prediction count:", len(errors))
print("\nMisclassified samples:")
print(errors.to_string(index=False))

# Compare the saved error rows with the cleaned dataset
print("\nCorresponding rows in cleaned dataset:")

for index in error_indices:
    if index in clean.index:
        print(f"\nCleaned CSV index: {index}")
        print(clean.loc[[index]].to_string(index=False))
    else:
        print(f"Index {index} not found in cleaned dataset index.")

# Check whether suspicious measurements are also present in the original file
print("\nOriginal Excel rows with these row numbers:")

for index in error_indices:
    print(f"\nExcel row index: {index}")
    if index in original.index:
        print(original.loc[[index]].to_string(index=False))
    else:
        print("This index does not exist in the original Excel file.")
