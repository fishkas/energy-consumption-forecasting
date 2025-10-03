import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

print("=" * 60)
print("🚀 ПРОГНОЗИРОВАНИЕ ЭНЕРГОПОТРЕБЛЕНИЯ ЗДАНИЙ")
print("=" * 60)

# Загрузка и предобработка данных
df = pd.read_csv('data/raw_data.csv')

# Создание новых признаков
df['building_age'] = 2024 - df['year_built']
df['energy_per_sqft'] = df['energy_consumption'] / df['square_footage']

# Кодирование категориальных переменных
le = LabelEncoder()
df['building_type_encoded'] = le.fit_transform(df['building_type'])
df['heating_type_encoded'] = le.fit_transform(df['heating_type'])

# Сохранение обработанных данных
df.to_csv('data/processed_data.csv', index=False)

# Обучение модели
features = ['square_footage', 'occupant_count', 'avg_temperature', 
            'avg_humidity', 'building_age', 'building_type_encoded', 'heating_type_encoded']
X = df[features]
y = df['energy_consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Важность признаков
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 📊 ВЫВОД РЕЗУЛЬТАТОВ
print("\n📈 РЕЗУЛЬТАТЫ МОДЕЛИ:")
print("-" * 40)
print(f"• Средняя абсолютная ошибка (MAE): {mae:.2f} кВт·ч")
print(f"• Коэффициент детерминации (R²): {r2:.4f}")
print(f"• Точность прогноза: {r2*100:.1f}%")

print("\n🔍 ВАЖНОСТЬ ФАКТОРОВ:")
print("-" * 40)
for i, row in importance.iterrows():
    print(f"• {row['feature']:25} - {row['importance']*100:5.1f}%")

print("\n💡 КЛЮЧЕВЫЕ ВЫВОДЫ:")
print("-" * 40)
top_feature = importance.iloc[0]['feature']
top_importance = importance.iloc[0]['importance'] * 100

print(f"• Главный фактор влияния: {top_feature} ({top_importance:.1f}%)")

if top_feature == 'square_footage':
    print("• Рекомендация: Оптимизируйте энергопотребление в зданиях с большой площадью")
elif top_feature == 'occupant_count':
    print("• Рекомендация: Внедрите системы контроля потребления в зданиях с высокой населенностью")
elif top_feature == 'building_type_encoded':
    print("• Рекомендация: Разработайте отдельные стратегии для коммерческих и жилых зданий")
elif top_feature == 'avg_temperature':
    print("• Рекомендация: Улучшите теплоизоляцию и системы климат-контроля")

print("\n" + "=" * 60)
print("✅ ПРОЕКТ ЗАВЕРШЕН")
print("=" * 60)

# ⭐ ДОБАВЛЕНО: Ожидание нажатия клавиши
print("\nНажмите Enter для выхода...")
# Сохранение результатов в файл
with open('reports/results.txt', 'w', encoding='utf-8') as f:
    f.write("РЕЗУЛЬТАТЫ АНАЛИЗА ЭНЕРГОПОТРЕБЛЕНИЯ\n")
    f.write("=" * 50 + "\n")
    f.write(f"Точность модели: {r2*100:.1f}%\n")
    f.write(f"Средняя ошибка: {mae:.2f} кВт·ч\n")
    f.write("\nТоп-3 фактора влияния:\n")
    for i, row in importance.head(3).iterrows():
        f.write(f"{i+1}. {row['feature']} - {row['importance']*100:.1f}%\n")

print("📄 Результаты также сохранены в файл: reports/results.txt")
input()
