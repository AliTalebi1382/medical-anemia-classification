
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd


# مسیرهای پروژه
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"

# بارگذاری مدل آموزش‌دیده
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ساخت پنجره اصلی
root = tk.Tk()
root.title("Anemia Classification System")
root.geometry("560x650")
root.configure(bg="#F2F6FC")
root.resizable(False, False)

# عنوان برنامه
title = tk.Label(
    root,
    text="Anemia Classification",
    font=("Arial", 20, "bold"),
    bg="#F2F6FC",
    fg="#17365D"
)
title.pack(pady=(20, 5))

subtitle = tk.Label(
    root,
    text="Enter the hematological parameters",
    font=("Arial", 10),
    bg="#F2F6FC",
    fg="#52657A"
)
subtitle.pack(pady=(0, 15))


# فریم ورودی‌ها
form = tk.Frame(root, bg="#F2F6FC")
form.pack(padx=35, fill="x")

fields = [
    ("Gender (F/M)", "Gender"),
    ("Age", "Age"),
    ("Hemoglobin (HGB)", "HGB(Hemoglobin)"),
    ("Red Blood Cells (RBC)", "RBC"),
    ("Hematocrit (PCV/HCT)", "PCV/HCT"),
    ("MCV", "MCV"),
    ("MCH", "MCH"),
    ("MCHC", "MCHC"),
]

entries = {}

for row, (label_text, field_name) in enumerate(fields):
    label = tk.Label(
        form,
        text=label_text,
        font=("Arial", 10),
        bg="#F2F6FC",
        anchor="w",
        width=25
    )
    label.grid(row=row, column=0, padx=5, pady=6, sticky="w")

    if field_name == "Gender":
        entry = ttk.Combobox(
            form,
            values=["F", "M"],
            state="readonly",
            width=23
        )
        entry.set("F")
    else:
        entry = ttk.Entry(form, width=26)

    entry.grid(row=row, column=1, padx=5, pady=6)
    entries[field_name] = entry


def predict():
    try:
        gender = entries["Gender"].get().strip().upper()

        if gender not in ["F", "M"]:
            messagebox.showerror(
                "Input Error",
                "Select F or M for Gender."
            )
            return

        values = {
            "Gender": 0 if gender == "F" else 1
        }

        numeric_fields = [
            "Age",
            "HGB(Hemoglobin)",
            "RBC",
            "PCV/HCT",
            "MCV",
            "MCH",
            "MCHC",
        ]

        for field in numeric_fields:
            text_value = entries[field].get().strip()

            if not text_value:
                messagebox.showerror(
                    "Input Error",
                    f"Please enter {field}."
                )
                return

            values[field] = float(text_value)

        # ترتیب ستون‌ها باید با زمان آموزش مدل یکسان باشد
        feature_order = [
            "Gender",
            "Age",
            "HGB(Hemoglobin)",
            "RBC",
            "PCV/HCT",
            "MCV",
            "MCH",
            "MCHC",
        ]

        input_data = pd.DataFrame(
            [[values[field] for field in feature_order]],
            columns=feature_order
        )

        prediction = int(model.predict(input_data)[0])

        # نمایش کلاس پیش‌بینی‌شده
        result_text = f"Predicted class: {prediction}"

        # نمایش احتمال پیش‌بینی‌شده برای هر کلاس
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            result_text += "\n\nPredicted probabilities:"

            for class_label, probability in zip(classes, probabilities):
                result_text += (
                    f"\nClass {class_label}: "
                    f"{probability * 100:.2f}%"
                )

        result_text += (
            "\n\nWarning: Educational use only. "
            "This model is not a medical diagnostic tool."
        )

        messagebox.showinfo("Prediction Result", result_text)

    except ValueError:
        messagebox.showerror(
            "Input Error",
            "Enter valid numeric values in all measurement fields."
        )
    except Exception as error:
        messagebox.showerror(
            "Prediction Error",
            str(error)
        )


# دکمه پیش‌بینی
predict_button = tk.Button(
    root,
    text="Predict",
    command=predict,
    font=("Arial", 12, "bold"),
    bg="#1769AA",
    fg="white",
    activebackground="#12558A",
    activeforeground="white",
    width=18,
    relief="flat",
    cursor="hand2"
)
predict_button.pack(pady=22)

# هشدار پزشکی
warning = tk.Label(
    root,
    text="For educational use only — not for clinical decisions.",
    font=("Arial", 9),
    bg="#F2F6FC",
    fg="#A33A3A"
)
warning.pack(pady=5)

root.mainloop()
