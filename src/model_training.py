import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("=== ОБУЧЕНИЕ МОДЕЛИ ===")

# Загрузка данных
df = pd.read_csv('../data/processed_data.csv')

# Признаки и целевая переменная
features = ['square_footage', 'occupant_count', 'avg_temperature', 
            'building_age', 'building_type_encoded', 'heating_type_encoded']
X = df[features]
y = df['energy_consumption']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Оценка модели
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nРезультаты модели:")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Важность признаков
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nВажность признаков:")
print(importance)

print("\nОбучение завершено!")