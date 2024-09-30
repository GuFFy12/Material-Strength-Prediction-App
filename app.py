import joblib
import pandas as pd
import json
import numpy as np
import tkinter as tk
from tkinter import messagebox

model = joblib.load('random_forest_model.joblib')
feature_columns = joblib.load('model_features.joblib')

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    categories = sorted({entry['category'] for entry in data})
    rollings = sorted({entry['rolling'] for entry in data})

def make_prediction():
    category = category_var.get() if category_var.get() != 'None' else None
    rolling = rolling_var.get() if rolling_var.get() != 'None' else None

    try:
        size = float(size_entry.get()) if size_entry.get().strip() else np.nan
    except ValueError:
        messagebox.showerror("Invalid Input", "Size must be a number.")
        return

    composition = {}
    composition_text = composition_entry.get("1.0", tk.END).strip()

    if composition_text:
        lines = composition_text.splitlines()
        for line in lines:
            try:
                element, value = line.split(":")
                element = element.strip().lower()
                value = float(value.strip())
                composition[element] = value
            except ValueError:
                messagebox.showerror("Invalid Input",
                                     f"Invalid composition format in line: '{line}'. Use 'Element: value' format.")
                return

    input_data = pd.DataFrame([{col: 0 for col in feature_columns}])
    input_data = input_data.astype(float)

    for element, value in composition.items():
        if element in input_data.columns:
            input_data.at[0, element] = value

    if 'size' in input_data.columns and not np.isnan(size):
        input_data.at[0, 'size'] = size

    if category:
        cat_col = f'category_{category}'
        if cat_col in input_data.columns:
            input_data.at[0, cat_col] = 1.0

    if rolling:
        roll_col = f'rolling_{rolling}'
        if roll_col in input_data.columns:
            input_data.at[0, roll_col] = 1.0

    input_data = input_data.dropna(axis=1)

    input_data = input_data.reindex(columns=[col for col in feature_columns if col in input_data.columns], fill_value=0)

    try:
        prediction = model.predict(input_data)
        messagebox.showinfo("Prediction Result", f"Predicted Ultimate Tensile Strength (σ_U): {prediction[0]:.2f} MPa")
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))


root = tk.Tk()
root.title("Material Strength Prediction App")

tk.Label(root, text="Select Category:").grid(row=0, column=0, sticky='e')
category_var = tk.StringVar(value='None')
category_menu = tk.OptionMenu(root, category_var, 'None', *categories)
category_menu.grid(row=0, column=1)

tk.Label(root, text="Select Rolling Type:").grid(row=1, column=0, sticky='e')
rolling_var = tk.StringVar(value='None')
rolling_menu = tk.OptionMenu(root, rolling_var, 'None', *rollings)
rolling_menu.grid(row=1, column=1)

tk.Label(root, text="Size (mm):").grid(row=2, column=0, sticky='e')
size_entry = tk.Entry(root)
size_entry.grid(row=2, column=1)

tk.Label(root, text="Composition (Element: Value per line):").grid(row=3, column=0, sticky='ne')
composition_entry = tk.Text(root, height=10, width=30)
composition_entry.grid(row=3, column=1)

predict_button = tk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=4, columnspan=2)

root.mainloop()
