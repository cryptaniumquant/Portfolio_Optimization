import os
import pandas as pd
import quantstats as qs
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Подавляем предупреждения
warnings.filterwarnings('ignore')

# Загружаем данные стратегий
data_dir = r'c:\Users\Ilya3\AlgoYES\Cryptanium_work_algo_opt\data_prep\adjusted_data\grid\complete'
all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Создаем словарь для хранения данных каждой стратегии
strategies_data = {}

# Загружаем данные каждой стратегии
for file in all_files:
    strategy_name = file.split('_')[1]  # Получаем название стратегии из имени файла
    df = pd.read_csv(os.path.join(data_dir, file), index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Объединяем backtest и real значения
    combined_values = df['backtest_Value'].fillna(df['real_Value'])
    
    # Находим точку перехода между бэктестом и реальными данными
    transition_point = df[df['real_Value'].notna()].index[0]
    print(f"{strategy_name}: Transition at {transition_point}")
    
    # Сохраняем данные в словарь
    strategies_data[strategy_name] = combined_values

# Создаем DataFrame со всеми стратегиями
portfolio_df = pd.DataFrame(strategies_data)

# Определяем периоды обучения и тестирования
training_end = '2025-01-01'
print(f"\nTraining period: {portfolio_df.index[0]} to {training_end}")
print(f"Test period: {pd.to_datetime(training_end) + pd.Timedelta(days=1)} to {portfolio_df.index[-1]}")

# Создаем директорию для отчетов
current_dir = os.path.dirname(os.path.abspath(__file__))
report_dir = os.path.join(current_dir, 'reports')
os.makedirs(report_dir, exist_ok=True)


# --- User weights (сумма = 100) ---
user_weights = {
    'AaveUsdt': 7.02*0.75+0.25*1,
    'AdaUsdt': 5.99*0.75+0.25*1,
    'AtomUsdt': 6.23*0.75+0.25*15,
    'BchUsdt': 8.98*0.75+0.25*15,
    'BnbUsdt': 13.18*0.75+0.25*1,
    'BtcUsdt': 13.36*0.75+0.25*15,
    'CompUsdt': 5.00*0.75+0.25*1,
    'DogeUsdt': 6.70*0.75+0.25*13.5,
    'DydxUsdt': 1.85*0.75+0.25*1,
    'FilUsdt': 2.78*0.75+0.25*15,
    'IcpUsdt': 3.93*0.75+0.25*1,
    'LinkUsdt': 7.36*0.75+0.25*8.1,
    'LtcUsdt': 8.77*0.75+0.25*1,
    'RoseUsdt': 3.67*0.75+0.25*1,
    'StxUsdt': 1.34*0.75+0.25*9.2,
    'SuiUsdt': 3.85*0.75+0.25*1
}
# Нормируем веса на 1
user_weights_norm = {k: v / 100.0 for k, v in user_weights.items()}

# Проверяем, что сумма весов равна 1 (с учетом погрешности округления)
weights_sum = sum(user_weights_norm.values())
if not 0.9999 <= weights_sum <= 1.0001:  # Допускаем небольшую погрешность округления
    print(f"WARNING: Sum of normalized weights is {weights_sum:.6f}, expected 1.0")
    # Нормализуем веса, чтобы их сумма была точно равна 1
    user_weights_norm = {k: v / weights_sum for k, v in user_weights_norm.items()}
    print("Weights have been renormalized to sum to 1.0")

# Выводим статистику по пользовательским весам
print("\nNormalized Strategy Weights:")
for symbol, weight in user_weights_norm.items():
    print(f"{symbol}: {weight:.2f}")

# --- Рассчитываем PnL равновзвешенного портфеля ---
equal_weight = 1.0 / len(portfolio_df.columns)
equal_weighted_pnl = pd.Series(0, index=portfolio_df.index)
for column in portfolio_df.columns:
    equal_weighted_pnl += portfolio_df[column] * equal_weight

# --- Рассчитываем PnL портфеля с пользовательскими весами ---
user_weighted_pnl = pd.Series(0, index=portfolio_df.index)
for symbol, weight in user_weights_norm.items():
    if symbol in portfolio_df.columns:
        user_weighted_pnl += portfolio_df[symbol] * weight
    else:
        print(f'WARNING: {symbol} not found in portfolio_df.columns!')

# Выбираем только тестовый период
test_equal_weighted_pnl = equal_weighted_pnl[training_end:]
test_user_weighted_pnl = user_weighted_pnl[training_end:]

# Ресемплим данные в часовой таймфрейм
equal_weighted_hourly = test_equal_weighted_pnl.resample('1H').min().fillna(method='ffill')
user_weighted_hourly = test_user_weighted_pnl.resample('1H').min().fillna(method='ffill')

# Строим NAV на основе PnL: NAV = 1 + (PnL - PnL[0])/100
equal_weighted_nav = 1 + (equal_weighted_hourly - equal_weighted_hourly.iloc[0]) / 100
equal_weighted_returns = equal_weighted_nav.pct_change().fillna(0)

user_weighted_nav = 1 + (user_weighted_hourly - user_weighted_hourly.iloc[0]) / 100
user_weighted_returns = user_weighted_nav.pct_change().fillna(0)

# Создаем график сравнения PnL
plt.figure(figsize=(15, 7))
plt.plot(test_equal_weighted_pnl.index, test_equal_weighted_pnl.values - test_equal_weighted_pnl.values[0], label='Equal-Weighted Portfolio')
plt.plot(test_user_weighted_pnl.index, test_user_weighted_pnl.values - test_user_weighted_pnl.values[0], label='User Weights Portfolio')
plt.title('Portfolio Comparison (Test Period)')
plt.xlabel('Date')
plt.ylabel('PnL (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'portfolio_comparison.png'))
plt.close()

# --- Выводим основные метрики для всех портфелей ---
def print_metrics(nav, returns, name):
    print(f'\n{name} Metrics:')
    print(f'Final NAV: {nav.iloc[-1]:.4f}')
    print(f'Total Return: {(nav.iloc[-1] - 1) * 100:.2f}%')
    print(f'Max Drawdown: {qs.stats.max_drawdown(nav):.2%}')
    print(f'Sharpe Ratio: {qs.stats.sharpe(returns):.2f}')

print_metrics(equal_weighted_nav, equal_weighted_returns, 'Equal-Weighted Portfolio')
print_metrics(user_weighted_nav, user_weighted_returns, 'User Weights Portfolio')

# --- Plotly drawdown comparison ---
# Calculate drawdowns
def calculate_drawdown(series):
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return drawdown

equal_weighted_drawdown = calculate_drawdown(equal_weighted_nav)
user_weighted_drawdown = calculate_drawdown(user_weighted_nav)

fig = go.Figure()
fig.add_trace(go.Scatter(x=equal_weighted_drawdown.index, y=equal_weighted_drawdown, mode='lines', name='Equal-Weighted Portfolio'))
fig.add_trace(go.Scatter(x=user_weighted_drawdown.index, y=user_weighted_drawdown, mode='lines', name='User Weights Portfolio'))
fig.update_layout(
    title='Drawdowns Comparison (Test Period)',
    xaxis_title='Date',
    yaxis_title='Drawdown',
    legend=dict(x=0.01, y=0.99),
    template='plotly_white',
    width=1000, height=500
)
fig.write_html(os.path.join(report_dir, 'drawdowns_comparison.html'))
print('Drawdowns comparison chart saved to:', os.path.join(report_dir, 'drawdowns_comparison.html'))

# Генерируем сравнительный отчет QuantStats
qs.extend_pandas()
if (user_weighted_nav != user_weighted_nav.iloc[0]).any():
    qs.reports.html(
        returns=user_weighted_returns,
        benchmark=equal_weighted_returns,
        benchmark_title='Equal-Weighted Portfolio',
        output=os.path.join(report_dir, 'portfolio_comparison_report.html'),
        title='User Weights vs Equal-Weighted Portfolio Analysis',
        periods_per_year=24*365
    )
    print('QuantStats report saved to:', os.path.join(report_dir, 'portfolio_comparison_report.html'))
else:
    print('User Weights Portfolio NAV is constant — QuantStats report not generated.')

# Save portfolio performance data to CSV
portfolio_data = pd.DataFrame({
    'Date': equal_weighted_nav.index,
    # 'Equal_Weighted_NAV': equal_weighted_nav.values,
    # 'User_Weighted_NAV': user_weighted_nav.values,
    'Equal_Weighted_Returns': equal_weighted_returns.values,
    'User_Weighted_Returns': user_weighted_returns.values,

})

# Set Date as index for the CSV
portfolio_data = portfolio_data.set_index('Date')

# Save to CSV
performance_csv_path = os.path.join(report_dir, 'portfolio_performance.csv')
portfolio_data.to_csv(performance_csv_path, float_format='%.6f')
print(f'Portfolio performance data saved to: {performance_csv_path}')
