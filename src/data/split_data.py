import pandas as pd
import glob
import os
import sys

def main():
    # Читаем все файлы
    files = glob.glob(os.path.join(folder, '*.csv'))
    
    print(f"Найдено файлов: {len(files)}")
    
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    # Преобразуем время
    df['click_timestamp'] = pd.to_datetime(df['click_timestamp'], unit='ms')

    # Сортируем по времени
    df = df.sort_values('click_timestamp').reset_index(drop=True)

    # Разделяем 80/20
    split_idx = int(len(df) * 0.8)
    train = df[:split_idx]
    test = df[split_idx:]

    # Сохраняем
    train.to_csv('data/processed/split_only_positive/train_data.csv', index=False)
    test.to_csv('data/processed/split_only_positive/test_data.csv', index=False)

    print(f"\nВсего: {len(df):,} записей")
    print(f"Train: {len(train):,} записей ({len(train)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test):,} записей ({len(test)/len(df)*100:.1f}%)")

if __name__ == '__main__':
    main()