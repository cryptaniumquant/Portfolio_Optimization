import os
import pandas as pd
import numpy as np

def ulcer_index(series: pd.Series, window: int = 30) -> pd.Series:
    # Вычисляет Ulcer Index по скользящему окну
    rolling_max = series.rolling(window, min_periods=1).max()
    drawdown = 100 * (series - rolling_max) / rolling_max
    squared_dd = drawdown ** 2
    ui = squared_dd.rolling(window, min_periods=1).mean().apply(np.sqrt)
    return ui

def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Агрегация часовых свечей в дневные
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    daily = df.resample('1D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    })
    daily = daily.dropna(subset=['Open', 'High', 'Low', 'Close'])
    return daily

def process_file(filepath: str) -> float:
    df = pd.read_csv(filepath)
    daily = aggregate_to_daily(df)
    close = daily['Close']
    ui = ulcer_index(close, window=30)
    return ui.median()

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'grid_price')
    results = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        symbol = fname.split('_')[0]
        fpath = os.path.join(data_dir, fname)
        try:
            avg_ui = process_file(fpath)
            results[symbol] = avg_ui
            print(f'{symbol}: {avg_ui:.4f}')
        except Exception as e:
            print(f'Error processing {fname}: {e}')
    # Скалирование значений: максимум -> 10, минимум -> 2
    vals = np.array(list(results.values()))
    if len(vals) > 1:
        min_v, max_v = vals.min(), vals.max()
        scaled = {k: 10 - 9 * (v - min_v) / (max_v - min_v) if max_v > min_v else 6 for k, v in results.items()}
    else:
        scaled = {k: 6 for k in results}  # если только одна монета

    print('\nМедианный Ulcer Index по монетам:')
    for k, v in results.items():
        print(f'{k}: {v:.4f}')
    print('\nСкалированные значения (от 10 до 2):')
    for k, v in scaled.items():
        print(f'{k}: {v:.2f}')

    # Нормировка: сумма = 100
    total = sum(scaled.values())
    if total > 0:
        normalized = {k: v * 100 / total for k, v in scaled.items()}
    else:
        normalized = {k: 0 for k in scaled}
    print('\nНормированные значения (сумма = 100):')
    for k, v in normalized.items():
        print(f'{k}: {v:.2f}')
    print(f'Сумма: {sum(normalized.values()):.2f}')

if __name__ == '__main__':
    main()
