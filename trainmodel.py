import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import joblib


df = pd.read_csv('train.csv')


TARGET_COL = 'responder_6'
WEIGHT_COL = 'weight'
TIME_COL = 'timestamp'
EXCLUDE_COLS = [TARGET_COL, WEIGHT_COL, TIME_COL]  # add more if needed

X = df.drop(columns=EXCLUDE_COLS)
y = df[TARGET_COL]
w = df[WEIGHT_COL]

#Define Custom Weighted Zero-Mean R² - kaggle competiton requirement!! *important*
def weighted_zero_mean_r2(y_true, y_pred, weights):
    y_true_mean = np.average(y_true, weights=weights)
    y_pred_mean = np.average(y_pred, weights=weights)
    
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean
    
    ss_res = np.sum(weights * (y_true_centered - y_pred_centered) ** 2)
    ss_tot = np.sum(weights * y_true_centered ** 2)
    
    return 1 - ss_res / ss_tot

#Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    ))
])

#Time-Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    w_test = w.iloc[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    score = weighted_zero_mean_r2(y_test.values, y_pred, w_test.values)
    scores.append(score)
    print(f"Fold {fold + 1} Weighted Zero-Mean R²: {score:.4f}")

print(f"\nAverage CV Score: {np.mean(scores):.4f}")

pipeline.fit(X, y)

#submission
joblib.dump(pipeline, 'model.pkl')
print("✅ Model saved to 'model.pkl'")
