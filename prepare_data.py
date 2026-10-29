
from pathlib import Path
import pandas as pd

# مسیر فایل اصلی اکسل
base_dir = Path(__file__).parent
input_path = (
    base_dir
    / "data"
    / "Raw Hematological Dataset for Anemia Analysis and Classification.xlsx"
)

# مسیر ذخیره دیتاست پاک‌سازی‌شده
output_path = base_dir / "data" / "anemia_clean.csv"

# خواندن داده‌ها
df = pd.read_excel(input_path)

# پاک‌سازی نام ستون‌ها و مقادیر جنسیت
df.columns = df.columns.str.strip()
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
target = "Decision_Class"

print("Original rows:", len(df))

# حذف ردیف‌های کاملاً تکراری
df = df.drop_duplicates().copy()
print("After removing exact duplicates:", len(df))

# حذف ردیف‌هایی که مقادیر ورودی یکسان ولی برچسب متفاوت دارند
label_counts = (
    df.groupby(features, dropna=False)[target]
    .transform("nunique")
)

conflict_mask = label_counts > 1
print("Rows with conflicting labels:", int(conflict_mask.sum()))

df = df.loc[~conflict_mask].copy()

# بررسی مقادیر گمشده در ورودی‌ها و ستون هدف
missing_input_rows = int(
    df[features].isna().any(axis=1).sum()
)

missing_target_rows = int(df[target].isna().sum())

print("Rows with missing input values:", missing_input_rows)
print("Rows with missing target values:", missing_target_rows)

# حذف ردیف‌های ناقص در ورودی یا برچسب
df = df.dropna(subset=features + [target]).copy()

# ذخیره فایل جدید؛ فایل اکسل اصلی دست‌نخورده می‌ماند
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\nCleaned dataset saved to:", output_path)
print("Final rows:", len(df))
print("Final columns:", list(df.columns))
print("\nClass distribution:")
print(df[target].value_counts().sort_index())
print("\nRemaining missing values:", int(df.isna().sum().sum()))
print("\nRemaining exact duplicates:", int(df.duplicated().sum()))
