# block#1
seed = 5
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Функции для загрузки и обработки данных ---
def load_grid_strategy_data(grid_data_dir, use_backtest=True, resample_freq=None, combine_columns=True):
    """
    Load grid strategy data from CSV files in the specified directory.
    Returns: DataFrame with all grid strategies
    """
    print(f"Loading grid strategy data from: {grid_data_dir}")
    csv_files = glob.glob(os.path.join(grid_data_dir, '*.csv'))
    csv_files = [f for f in csv_files if 'strategies_info' not in f]
    strategy_files = {}
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if 'combined' in file_name:
            parts = file_name.split('_combined_')
            strategy_name = parts[0]
            timestamp = parts[1].replace('.csv', '')
            if strategy_name not in strategy_files or timestamp > strategy_files[strategy_name][1]:
                strategy_files[strategy_name] = (file_path, timestamp)
    latest_files = [info[0] for info in strategy_files.values()]
    print(f"Found {len(latest_files)} latest grid strategy files")
    all_strategies = {}
    for file_path in latest_files:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        strategy_name = os.path.basename(file_path).split('_combined_')[0]
        has_backtest = 'backtest_Value' in df.columns
        has_real = 'real_Value' in df.columns
        if not has_backtest and not has_real:
            print(f"Warning: Neither backtest_Value nor real_Value found in {file_path}, skipping")
            continue
        if combine_columns and has_backtest and has_real:
            backtest_values = df['backtest_Value'].fillna(0)
            real_values = df['real_Value'].fillna(0)
            has_backtest_data = ~df['backtest_Value'].isna()
            has_real_data = ~df['real_Value'].isna()
            if use_backtest:
                combined_values = backtest_values.where(has_backtest_data, real_values)
            else:
                combined_values = real_values.where(has_real_data, backtest_values)
            print(f"{strategy_name}: Combined {has_backtest_data.sum()} backtest and {has_real_data.sum()} real values")
            all_strategies[strategy_name] = combined_values
        else:
            if use_backtest and has_backtest:
                all_strategies[strategy_name] = df['backtest_Value'].fillna(0)
                print(f"{strategy_name}: Using backtest_Value only")
            elif not use_backtest and has_real:
                all_strategies[strategy_name] = df['real_Value'].fillna(0)
                print(f"{strategy_name}: Using real_Value only")
            elif has_backtest:
                all_strategies[strategy_name] = df['backtest_Value'].fillna(0)
                print(f"{strategy_name}: Falling back to backtest_Value")
            elif has_real:
                all_strategies[strategy_name] = df['real_Value'].fillna(0)
                print(f"{strategy_name}: Falling back to real_Value")
    strategies_df = pd.DataFrame(all_strategies)
    if resample_freq is not None:
        strategies_df = strategies_df.resample(resample_freq).min().dropna()
        nan_count_postfill = strategies_df.isna().sum().sum()
        print(f"NaN remaining after ffill + bfill: {nan_count_postfill}")

    # 5. Итоговый диапазон
    print("Final min/max after fillna:", strategies_df.min().min(), strategies_df.max().max())
    
    # Check for any extreme values before resampling
    min_value = strategies_df.min().min()
    max_value = strategies_df.max().max()
    print(f"Data range before resampling: {min_value} to {max_value}")
    
    # Resample if requested
    if resample_freq is not None:
        print(f"Resampling data from {strategies_df.index[1] - strategies_df.index[0]} to '{resample_freq}' frequency")
        strategies_df = strategies_df.resample(resample_freq).min()  # Using min() for PnL values
        print(f"Data shape after resampling: {strategies_df.shape}")
    
    return strategies_df

# Function to calculate PnL changes (returns) from strategy values
def calculate_pnl_changes(df):
    """
    Calculate period-to-period changes in PnL values.
    
    Args:
        df: DataFrame with strategy PnL values
        
    Returns:
        DataFrame with strategy returns
    """
    print(f"Calculating returns from PnL values with shape {df.shape}")
    
    # For grid strategies, use absolute differences rather than percentage changes
    # This better reflects the actual returns from grid trading
    returns_df = df.diff()
    
    # Handle the first row (which contains NaN values after diff())
    returns_df = returns_df.fillna(0)

    # Replace inf/-inf with NaN, then fill NaN with -1e-6
    inf_count = np.isinf(returns_df.values).sum()
    nan_count = np.isnan(returns_df.values).sum()
    if inf_count > 0 or nan_count > 0:
        print(f"Found {inf_count} inf and {nan_count} NaN in returns after diff. Fixing...")
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
    returns_df = returns_df.fillna(-1e-6)

    # Verify results
    min_return = returns_df.min().min()
    max_return = returns_df.max().max()
    print(f"Return values range after cleaning: {min_return} to {max_return}")
    print(f"Any NaN left in returns? {returns_df.isna().any().any()}")
    print(f"Any inf left in returns? {np.isinf(returns_df.values).any()}")
    
        
    #     # Add some padding to the thresholds
    #     lower_bound = min(p01 * 1.5, -0.05)  # At least -0.05
    #     upper_bound = max(p99 * 1.5, 0.05)   # At least 0.05
    return returns_df

# Load grid strategy data from the specified directory
grid_data_dir = 'c:\\Users\\Ilya3\\AlgoYES\\Cryptanium_work_algo_opt\\data_prep\\adjusted_data\\grid\\complete'

# Use combined data (backtest + real) for most complete dataset
grid_strategies_df = load_grid_strategy_data(
    grid_data_dir, 
    use_backtest=False,  # Prioritize real_Value when available
    resample_freq='1H',  # Daily resampling
    combine_columns=True  # Combine backtest and real data
)

# Show a sample of the loaded data
print(f"\nFirst 3 rows of grid strategy data:")
print(grid_strategies_df.iloc[:3, :3])  # Show first 3 rows, first 3 columns

print(f"\nLast 3 rows of grid strategy data:")
print(grid_strategies_df.iloc[-3:, :3])  # Show last 3 rows, first 3 columns

# Calculate returns (PnL changes)
grid_returns_df = calculate_pnl_changes(grid_strategies_df)

# Check the calculated returns
print(f"\nFirst 3 rows of returns data:")
print(grid_returns_df.iloc[:3, :3])

print(f"\nLast 3 rows of returns data:")
print(grid_returns_df.iloc[-3:, :3])

# Use grid strategies for portfolio optimization
all_strategies_df = grid_returns_df

print(f"Loaded grid strategies: {grid_strategies_df.shape[1]}")
print(f"Time periods in data: {grid_strategies_df.shape[0]}")

# Risk-free asset has been removed from calculations
print("\nNo risk-free asset used in optimization")

# Fill missing values with zeros instead of dropping rows
all_strategies_df = all_strategies_df.fillna(0)
print(f"After filling NaNs with zeros, rows: {all_strategies_df.shape[0]}")

print(f"Total strategies for optimization: {all_strategies_df.shape[1]}")
print(f"Available time periods: {all_strategies_df.shape[0]}")

# Ensure all data is numeric
if not all_strategies_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
    print("WARNING: Not all columns contain numeric data!")
    # In case of issues, output data types
    print(all_strategies_df.dtypes)

# Force convert to numeric type
all_strategies_df = all_strategies_df.apply(pd.to_numeric, errors='coerce')

# Save unnormalized returns for statistics/plots
raw_returns_df = all_strategies_df.copy()

# Normalize returns: zero mean, unit variance per asset (for model only)


# Split data into training and test sets (80% train, 20% test)
train_size = int(len(all_strategies_df) * 0.85)
returns_train = all_strategies_df.iloc[:train_size]
returns_test = all_strategies_df.iloc[train_size:]
print(f"\nTraining set size: {len(returns_train)} periods")
print(f"Test set size: {len(returns_test)} periods")
print(f"Training period: {returns_train.index[0]} to {returns_train.index[-1]}")
print(f"Test period: {returns_test.index[0]} to {returns_test.index[-1]}")


def optimize_portfolio_wr(returns_train, allow_short=False, min_weight=0.0, max_weight=1.0):
    """
    Оптимизация портфеля по критерию Worst Realization (максимальный убыток).
    :param returns_train: DataFrame с доходностями (periods x assets)
    :param allow_short: разрешить короткие позиции (short) или нет
    :param min_weight: минимальный вес одной стратегии
    :param max_weight: максимальный вес одной стратегии
    :return: оптимальные веса (np.array)
    """
    import cvxpy as cp
    import numpy as np
    X = returns_train.values  # (T, N)
    n_assets = X.shape[1]
    w = cp.Variable(n_assets)
    portfolio_returns = X @ w  # shape (T,)
    wr = cp.max(-portfolio_returns)  # максимальный убыток

    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints += [w >= min_weight]
        constraints += [w <= max_weight]
    # Если разрешить short, ограничения можно скорректировать по желанию

    prob = cp.Problem(cp.Minimize(wr), constraints)
    prob.solve(solver=cp.ECOS)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Optimization failed:", prob.status)
        return None

    return w.value

# --- Workflow: оптимизация по WR (Worst Realization) ---
min_weight = 0.01  # Например, не менее 2% на одну стратегию
max_weight = 0.15   # Например, не более 20% на одну стратегию
weights_wr = optimize_portfolio_wr(returns_train, min_weight=min_weight, max_weight=max_weight)
if weights_wr is not None:
    print("Optimal weights (WR):", weights_wr)
    print("Sum of weights:", np.sum(weights_wr))
    optimal_weights = weights_wr  # Для совместимости с остальным кодом
else:
    print("WR optimization failed!")
    optimal_weights = None


# Функция для визуализации данных
def plot_optimization_data(raw_df, normalized_df, train_size):
    # Создаем график с двумя подграфиками
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Raw Returns', 'Normalized Returns'),
        vertical_spacing=0.15
    )

    # Добавляем линии для каждой стратегии на первом графике (raw data)
    for col in raw_df.columns[:20]:  # Показываем первые 5 стратегий для наглядности
        fig.add_trace(
            go.Scatter(x=raw_df.index, y=raw_df[col], name=col, mode='lines', opacity=0.7),
            row=1, col=1
        )
    
    # Добавляем вертикальную линию, разделяющую train/test
    split_date = raw_df.index[train_size]
    # Добавляем shapes вместо vline для корректной работы с datetime
    fig.add_shape(
        type='line',
        x0=split_date,
        x1=split_date,
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='red', width=2, dash='dash'),
        row=1,
        col=1
    )
    fig.add_shape(
        type='line',
        x0=split_date,
        x1=split_date,
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='red', width=2, dash='dash'),
        row=2,
        col=1
    )
    
    # Добавляем аннотации для split line
    fig.add_annotation(
        x=split_date,
        y=1,
        text='Train/Test Split',
        showarrow=True,
        arrowhead=2,
        row=1,
        col=1
    )
    fig.add_annotation(
        x=split_date,
        y=1,
        text='Train/Test Split',
        showarrow=True,
        arrowhead=2,
        row=2,
        col=1
    )

    # Добавляем линии для каждой стратегии на втором графике (normalized data)
    for col in normalized_df.columns[:5]:
        fig.add_trace(
            go.Scatter(x=normalized_df.index, y=normalized_df[col], 
                      name=col + ' (norm)', mode='lines', opacity=0.7),
            row=2, col=1
        )

    # Обновляем layout
    fig.update_layout(
        height=800,
        title_text='Strategy Returns: Raw vs Normalized',
        showlegend=True,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=1.05
        )
    )

    # Сохраняем график
    fig.write_html('grid_strategy_optimization_main/optimization_data_visualization.html')
    print('График сохранен в файл optimization_data_visualization.html')

# Визуализируем данные до и после нормализации
plot_optimization_data(raw_returns_df, all_strategies_df, train_size)

# Save optimal weights to CSV
weights_df = pd.DataFrame({'strategy': returns_train.columns, 'weight': optimal_weights})
weights_df.to_csv('grid_strategy_optimization_main/optimal_strategy_weights.csv', index=False)

