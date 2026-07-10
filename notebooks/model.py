import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load data
df = pd.read_excel('../data/yield_df_with_area_and_nigeria.xlsx')
df = df.drop(columns=['Unnamed: 0'])
print("Loaded shape:", df.shape)

# 2. Encode categorical columns
le_area = LabelEncoder()
le_item = LabelEncoder()
df['Area_encoded'] = le_area.fit_transform(df['Area'])
df['Item_encoded'] = le_item.fit_transform(df['Item'])

# 3. Remove outliers (IQR) on target
Q1 = df['hg/ha_yield'].quantile(0.25)
Q3 = df['hg/ha_yield'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
before = df.shape[0]
df = df[(df['hg/ha_yield'] >= lower_bound) & (df['hg/ha_yield'] <= upper_bound)]
print(f"Outlier removal: {before} -> {df.shape[0]} rows")

# 4. Correlation analysis (before scaling, using raw values)
corr_cols = ['Area_encoded', 'Item_encoded', 'Year', 'average_rain_fall_mm_per_year',
             'pesticides_tonnes', 'avg_temp', 'area_harvested_ha', 'hg/ha_yield']
corr_matrix = df[corr_cols].corr()
print("\nCorrelation with yield:")
print(corr_matrix['hg/ha_yield'].sort_values(ascending=False))

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap of Features and Yield')
plt.tight_layout()
plt.savefig('../models/correlation_heatmap.png')
plt.close()

# 5. Min-Max Normalization
features_to_scale = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'area_harvested_ha']
scaler = MinMaxScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# 6. Train/test split
feature_cols = ['Area_encoded', 'Item_encoded', 'Year', 'average_rain_fall_mm_per_year',
                 'pesticides_tonnes', 'avg_temp', 'area_harvested_ha']
X = df[feature_cols]
y = df['hg/ha_yield']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['Item_encoded']
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# 7. Train models
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# 8. Evaluate
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)

xgb_r2 = r2_score(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print("\n=== Random Forest ===")
print(f"R2: {rf_r2:.4f}  RMSE: {rf_rmse:.2f}  MAE: {rf_mae:.2f}")
print("=== XGBoost ===")
print(f"R2: {xgb_r2:.4f}  RMSE: {xgb_rmse:.2f}  MAE: {xgb_mae:.2f}")

best_model = rf_model if rf_r2 > xgb_r2 else xgb_model
best_name = "Random Forest" if rf_r2 > xgb_r2 else "XGBoost"
best_preds = rf_preds if best_name == "Random Forest" else xgb_preds
print(f"\nBest model: {best_name}")

# 9. Feature importance
importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)

plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title(f'{best_name} Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../models/feature_importance.png')
plt.close()

# 10. Actual vs Predicted
plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_preds, alpha=0.4, s=15)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Yield (hg/ha)')
plt.ylabel('Predicted Yield (hg/ha)')
plt.title(f'Actual vs Predicted Yield - {best_name}')
plt.tight_layout()
plt.savefig('../models/actual_vs_predicted.png')
plt.close()

# 11. Model comparison chart
metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'R2': [rf_r2, xgb_r2],
    'RMSE': [rf_rmse, xgb_rmse],
    'MAE': [rf_mae, xgb_mae]
})
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric in zip(axes, ['R2', 'RMSE', 'MAE']):
    ax.bar(metrics_df['Model'], metrics_df[metric], color=['#4C72B0', '#DD8452'])
    ax.set_title(metric)
plt.tight_layout()
plt.savefig('../models/model_comparison.png')
plt.close()

# 12. Save model, encoders, scaler
os.makedirs('../models', exist_ok=True)
pickle.dump(best_model, open('../models/best_model.pkl', 'wb'))
pickle.dump(le_area, open('../models/label_encoder_area.pkl', 'wb'))
pickle.dump(le_item, open('../models/label_encoder_item.pkl', 'wb'))
pickle.dump(scaler, open('../models/scaler.pkl', 'wb'))

print("\nAll files saved to ../models/")
print("Done.")