
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define project paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "anemia_clean.csv"
RESULTS_PATH = BASE_DIR / "results" / "model_comparison.csv"
OUTPUT_DIR = BASE_DIR / "results"

# 2. Create results folder if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. Load dataset and model comparison results
df = pd.read_csv(DATA_PATH)
results = pd.read_csv(RESULTS_PATH)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()
results.columns = results.columns.str.strip()

# 4. Check required columns
required_columns = {"Model", "Accuracy", "F1-score"}
missing_columns = required_columns - set(results.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in model_comparison.csv: {missing_columns}\n"
        f"Available columns: {results.columns.tolist()}"
    )

print("Dataset shape:", df.shape)
print("Model results columns:", results.columns.tolist())

# 5. Chart 1: Distribution of classes
class_counts = (
    df["Decision_Class"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(7, 5))
class_counts.plot(kind="bar")
plt.title("Distribution of Decision Classes")
plt.xlabel("Decision Class")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

class_chart = OUTPUT_DIR / "class_distribution.png"
plt.savefig(class_chart, dpi=300)
plt.close()

# 6. Sort models by F1-score
results = results.sort_values(
    by="F1-score",
    ascending=False
)

# 7. Chart 2: Compare models by F1-score
plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["F1-score"])
plt.title("Model Comparison - F1 Score")
plt.xlabel("Machine Learning Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

f1_chart = OUTPUT_DIR / "model_f1_comparison.png"
plt.savefig(f1_chart, dpi=300)
plt.close()

# 8. Chart 3: Compare models by accuracy
plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["Accuracy"])
plt.title("Model Comparison - Accuracy")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

accuracy_chart = OUTPUT_DIR / "model_accuracy_comparison.png"
plt.savefig(accuracy_chart, dpi=300)
plt.close()

# 9. Print results
print("\nModel comparison:")
print(
    results[
        ["Model", "Accuracy", "F1-score", "CV F1 mean", "CV F1 std"]
    ].to_string(index=False)
)

print("\nCharts saved successfully:")
print(class_chart)
print(f1_chart)
print(accuracy_chart)
