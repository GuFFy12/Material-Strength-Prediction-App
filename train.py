import json
import pandas as pd
from sklearn.model_selection import train_test_split

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

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
from joblib import dump

param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 0.5, 1, 1.5, 2]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='r2',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print(f"Лучшие параметры: {best_params}")

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    **best_params
)

xgb_model.fit(X_train, y_train)

cv_scores_initial = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='r2')
print(f"\nCross-Validated R^2 Scores before feature selection: {cv_scores_initial}")
print(f"Mean Cross-Validated R^2 Score before feature selection: {cv_scores_initial.mean()}")

selector = SelectFromModel(xgb_model, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()]
print(f"Выбрано {len(selected_features)} признаков из {X_train.shape[1]}")

xgb_model.fit(X_train_selected, y_train)

cv_scores_selected = cross_val_score(xgb_model, X_train_selected, y_train, cv=10, scoring='r2')
print(f"\nCross-Validated R^2 Scores after feature selection: {cv_scores_selected}")
print(f"Mean Cross-Validated R^2 Score after feature selection: {cv_scores_selected.mean()}")

dump(xgb_model, 'final_xgb_model.joblib')
dump(selector, 'final_feature_selector.joblib')
print("Модель и селектор признаков успешно сохранены.")
