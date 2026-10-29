
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
from pathlib import Path

# بارگذاری مدل آموزش‌دیده
MODEL_PATH = Path(__file__).parent / "anemia_model.joblib"
model = joblib.load(MODEL_PATH)

# رنگ‌های رابط کاربری
BG = "#F4F7FB"
CARD = "#FFFFFF"
BLUE = "#176B87"
DARK = "#183153"
GREEN = "#16845B"
RED = "#C0392B"
GRAY = "#64748B"

# ساخت پنجره اصلی
root = tk.Tk()
root.title("سامانه هوشمند بررسی کم‌خونی")
root.geometry("520x680")
root.configure(bg=BG)
root.resizable(False, False)

# سربرگ
header = tk.Frame(root, bg=BLUE, height=110)
header.pack(fill="x")
header.pack_propagate(False)

tk.Label(
    header,
    text="سامانه هوشمند بررسی کم‌خونی",
    font=("Tahoma", 18, "bold"),
    bg=BLUE,
    fg="white"
).pack(pady=(22, 5))

tk.Label(
    header,
    text="پیش‌بینی آزمایش خون با یادگیری ماشین",
    font=("Tahoma", 10),
    bg=BLUE,
    fg="white"
).pack()

# کارت ورودی‌ها
card = tk.Frame(root, bg=CARD, padx=28, pady=18)
card.pack(fill="x", padx=25, pady=20)

tk.Label(
    card,
    text="اطلاعات آزمایش خون",
    font=("Tahoma", 13, "bold"),
    bg=CARD,
    fg=DARK
).pack(anchor="e", pady=(0, 12))

fields = [
    ("جنسیت (0 یا 1)", "gender"),
    ("هموگلوبین (Hemoglobin)", "hemoglobin"),
    ("MCH", "mch"),
    ("MCHC", "mchc"),
    ("MCV", "mcv")
]

entries = {}

for label_text, field_name in fields:
    row = tk.Frame(card, bg=CARD)
    row.pack(fill="x", pady=5)

    tk.Label(
        row,
        text=label_text,
        font=("Tahoma", 10),
        bg=CARD,
        fg=DARK,
        width=25,
        anchor="e"
    ).pack(side="right", padx=(5, 10))

    entry = tk.Entry(
        row,
        font=("Tahoma", 11),
        justify="center",
        width=15,
        relief="solid",
        bd=1
    )
    entry.pack(side="right", ipady=4)

    entries[field_name] = entry


def predict_anemia():
    try:
        gender = int(entries["gender"].get())
        hemoglobin = float(entries["hemoglobin"].get())
        mch = float(entries["mch"].get())
        mchc = float(entries["mchc"].get())
        mcv = float(entries["mcv"].get())

        if gender not in [0, 1]:
            messagebox.showerror(
                "خطای ورودی",
                "جنسیت باید مطابق کدگذاری دیتاست، 0 یا 1 باشد."
            )
            return

        values = [
            gender, hemoglobin, mch, mchc, mcv
        ]

        if not all(pd.notna(value) for value in values):
            messagebox.showerror(
                "خطای ورودی",
                "همه مقادیر باید عدد معتبر باشند."
            )
            return

        new_data = pd.DataFrame([{
            "Gender": gender,
            "Hemoglobin": hemoglobin,
            "MCH": mch,
            "MCHC": mchc,
            "MCV": mcv
        }])

        prediction = model.predict(new_data)[0]
        probabilities = model.predict_proba(new_data)[0]

        if prediction == 1:
            result_label.config(
                text="نتیجه مدل: کلاس کم‌خونی",
                fg=RED
            )
        else:
            result_label.config(
                text="نتیجه مدل: کلاس بدون کم‌خونی",
                fg=GREEN
            )

        probability_text = "\n".join(
            f"کلاس {label}: {probability:.2%}"
            for label, probability in zip(
                model.classes_, probabilities
            )
        )

        probability_label.config(
            text="احتمال‌های تخمینی مدل:\n" + probability_text
        )

    except ValueError:
        messagebox.showerror(
            "خطای ورودی",
            "لطفاً در تمام کادرها مقدار عددی معتبر وارد کنید."
        )


def clear_fields():
    for entry in entries.values():
        entry.delete(0, tk.END)

    result_label.config(
        text="نتیجه پیش‌بینی اینجا نمایش داده می‌شود",
        fg=DARK
    )
    probability_label.config(text="")


# دکمه‌ها
buttons = tk.Frame(root, bg=BG)
buttons.pack(pady=2)

tk.Button(
    buttons,
    text="پیش‌بینی",
    command=predict_anemia,
    font=("Tahoma", 11, "bold"),
    bg=BLUE,
    fg="white",
    activebackground="#12566D",
    activeforeground="white",
    width=16,
    relief="flat",
    cursor="hand2"
).pack(side="right", padx=6, ipady=6)

tk.Button(
    buttons,
    text="پاک کردن",
    command=clear_fields,
    font=("Tahoma", 11),
    bg="#E2E8F0",
    fg=DARK,
    width=12,
    relief="flat",
    cursor="hand2"
).pack(side="right", padx=6, ipady=6)

# بخش نتیجه
result_frame = tk.Frame(
    root,
    bg=CARD,
    padx=15,
    pady=15
)
result_frame.pack(fill="x", padx=25, pady=18)

result_label = tk.Label(
    result_frame,
    text="نتیجه پیش‌بینی اینجا نمایش داده می‌شود",
    font=("Tahoma", 12, "bold"),
    bg=CARD,
    fg=DARK
)
result_label.pack(pady=5)

probability_label = tk.Label(
    result_frame,
    text="",
    font=("Tahoma", 10),
    bg=CARD,
    fg=GRAY,
    justify="left"
)
probability_label.pack(pady=5)

# هشدار علمی
tk.Label(
    root,
    text="این ابزار آموزشی است و جایگزین تشخیص پزشک نیست.",
    font=("Tahoma", 9),
    bg=BG,
    fg=GRAY
).pack(pady=5)

root.mainloop()
