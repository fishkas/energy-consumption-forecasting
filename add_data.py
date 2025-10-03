import pandas as pd
import os

def add_new_building():
    print("🏢 ДОБАВЛЕНИЕ НОВОГО ЗДАНИЯ В ДАТАСЕТ")
    print("=" * 40)
    print()
    
    # Чтение существующих данных
    try:
        df = pd.read_csv('data/raw_data.csv')
        print(f"✅ Текущий датасет: {len(df)} зданий")
    except:
        df = pd.DataFrame()
        print("✅ Создан новый датасет")
    
    # Ввод данных
    print("Введите данные нового здания:")
    building_id = len(df) + 1 if not df.empty else 1
    
    building_type = input("Тип здания (Commercial/Residential): ")
    square_footage = int(input("Площадь (кв.футы): "))
    year_built = int(input("Год постройки: "))
    heating_type = input("Тип отопления (Electric/Gas): ")
    occupant_count = int(input("Количество жильцов: "))
    month = int(input("Месяц (1-12): "))
    avg_temperature = float(input("Средняя температура: "))
    avg_humidity = float(input("Средняя влажность (%): "))
    energy_consumption = int(input("Потребление энергии (кВт·ч): "))
    
    # Создание новой строки
    new_data = {
        'building_id': building_id,
        'building_type': building_type,
        'square_footage': square_footage,
        'year_built': year_built,
        'heating_type': heating_type,
        'occupant_count': occupant_count,
        'month': month,
        'avg_temperature': avg_temperature,
        'avg_humidity': avg_humidity,
        'energy_consumption': energy_consumption
    }
    
    # Добавление в датасет
    new_df = pd.DataFrame([new_data])
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Сохранение
    df.to_csv('data/raw_data.csv', index=False)
    print()
    print(f"✅ Добавлено новое здание! Всего зданий: {len(df)}")
    print("Запустите run_project.py для пересчета модели")

if __name__ == "__main__":
    add_new_building()