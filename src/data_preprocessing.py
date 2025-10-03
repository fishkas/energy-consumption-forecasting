import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("=== ПРЕДОБРАБОТКА ДАННЫХ ===")

# Загрузка данных
df = pd.read_csv('../data/raw_data.csv')

print("Исходные данные:")
print(f"Размер: {df.shape}")
print(f"Пропуски:\n{df.isnull().sum()}")

# Создание новых признаков
df['building_age'] = 2024 - df['year_built']
df['energy_per_sqft'] = df['energy_consumption'] / df['square_footage']

# Кодирование категориальных переменных
le = LabelEncoder()
df['building_type_encoded'] = le.fit_transform(df['building_type'])
df['heating_type_encoded'] = le.fit_transform(df['heating_type'])

# Сохранение обработанных данных
df.to_csv('../data/processed_data.csv', index=False)

print("\nПосле обработки:")
print(f"Новые колонки: {list(df.columns)}")
print("Предобработка завершена!")