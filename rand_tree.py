import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []
for entry in data:
    flat_entry = {
        'url': entry['url'],
        'category': entry['category'],
        'rolling': entry['rolling'],
        'size': entry['size'],
        'sigma_u': entry['sigma_u']
    }

    for element, value in entry['composition'].items():
        flat_entry[element] = value

    rows.append(flat_entry)

df = pd.DataFrame(rows)
df = pd.get_dummies(df, columns=['category', 'rolling'], drop_first=True)

X = df.drop(columns=['url', 'sigma_u'])
y = df['sigma_u']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42, n_estimators=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print("Feature Importances:")
print(feature_importances.sort_values(ascending=False))