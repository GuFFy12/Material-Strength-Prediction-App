import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []
for entry in data:
    flat_entry = {
        'category': entry['category'],
        'rolling': entry['rolling'],
        'size': entry['size'],
        'sigma': entry['sigma']
    }

    for element, value in entry['composition'].items():
        flat_entry[element] = value

    rows.append(flat_entry)

df = pd.DataFrame(rows)
df = pd.get_dummies(df, columns=['category', 'rolling'], drop_first=True)

X = df.drop(columns=['sigma'])
y = df['sigma']

model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X, y)

joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(X.columns, 'model_features.joblib')

print("Model training completed and saved.")
