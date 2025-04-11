import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def task1():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv('opsd_raw.csv')
    required_columns = [
        'utc_timestamp',
        'DK_load_actual_entsoe_transparency',
        'DK_wind_generation_actual',
        'DK_solar_generation_actual'
    ]
    df = df[required_columns]

    print("=== Data Inspection ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())

    df = df.dropna()

    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df = df.sort_values('utc_timestamp')
    df['date'] = df['utc_timestamp'].dt.date
    df['season'] = df['utc_timestamp'].apply(get_season)
    daily_data = []
    for date, group in df.groupby('date'):
        if len(group) == 24:  # Только полные дни
            daily_data.append({
                'date': date,
                'season': group['season'].iloc[0],
                'load_24h': group['DK_load_actual_entsoe_transparency'].values,
                'wind_24h': group['DK_wind_generation_actual'].values,
                'solar_24h': group['DK_solar_generation_actual'].values
            })

    daily_df = pd.DataFrame(daily_data)
    print()
    print("days")
    print(len(daily_df))
    print()
    print("\n=== 5 Sample Days ===")
    for i in range(5):
        print(f"\nDate: {daily_df['date'].iloc[i]}, Season: {daily_df['season'].iloc[i]}")
        print(f"Load (first 5h): {daily_df['load_24h'].iloc[i]}")
        print(f"Wind (first 5h): {daily_df['wind_24h'].iloc[i]}")
        print(f"Solar (first 5h): {daily_df['solar_24h'].iloc[i]}")

    print("\n=== Season Distribution ===")
    print(daily_df['season'].value_counts())
    train_df, temp_df = train_test_split(daily_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    features = ['load_24h', 'wind_24h', 'solar_24h']
    scaler = StandardScaler()
    scalers = {
        'load': StandardScaler(),
        'wind': StandardScaler(),
        'solar': StandardScaler()
    }
    train_df_scaled = scale_data(train_df,scalers ,fit=True)  # Только здесь fit!
    val_df_scaled = scale_data(val_df, scalers)
    test_df_scaled = scale_data(test_df, scalers)
    for feature in ['load', 'wind', 'solar']:
        data = np.vstack(train_df_scaled[f'{feature}_24h'])
        print(f"{feature} - mean: {data.mean():.2f}, std: {data.std():.2f}")
    plt.figure(figsize=(18, 12))
    seasons = ['winter', 'spring', 'summer', 'autumn']

    for i, season in enumerate(seasons, 1):
        # Берем первый попавшийся день этого сезона из train данных
        season_data = train_df_scaled[train_df_scaled['season'] == season].iloc[0]

        # Создаем подграфик
        plt.subplot(2, 2, i)

        # Рисуем все три кривые
        plt.plot(season_data['load_24h'], label='Load', color='blue')
        plt.plot(season_data['wind_24h'], label='Wind', color='green')
        plt.plot(season_data['solar_24h'], label='Solar', color='orange')

        # Настройки графика
        plt.title(f'{season.capitalize()} (Scaled)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Scaled Value')
        plt.axhline(y=0, color='gray', linestyle='--')  # Нулевая линия
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    seasons_analysis = train_df_scaled.groupby('season').agg({
        'solar_24h': lambda x: np.mean(np.vstack(x), axis=0)
    })

    plt.figure(figsize=(10, 6))
    for season in seasons_analysis.index:
        plt.plot(seasons_analysis.loc[season, 'solar_24h'], label=season)
    plt.title('Average Solar Generation by Season')
    plt.legend()
    plt.show()
    return train_df_scaled, val_df_scaled, test_df_scaled


def scale_data(df, scalers, fit=False):
    scaled = df.copy()
    for feature, scaler in scalers.items():
        # Преобразуем список 24-часовых значений в 2D массив (n_days, 24)
        data = np.vstack(scaled[f'{feature}_24h'])

        if fit:
            # Только для train - обучаем scaler
            scaled_data = scaler.fit_transform(data)
        else:
            # Для val/test - только transform
            scaled_data = scaler.transform(data)

        # Преобразуем обратно в список списков
        scaled[f'{feature}_24h'] = [row.tolist() for row in scaled_data]
    return scaled



def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'spring'
    elif 6 <= month <= 8:
        return 'summer'
    elif 9 <= month <= 11:
        return 'autumn'
    else:
        return 'winter'

task1()