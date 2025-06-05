import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs
from datetime import datetime

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths
trend_file = r'c:\Users\Ilya3\AlgoYES\Cryptanium_work_algo_opt\Final_table_main\trend_combined_equal_weight_returns.csv'
grid_file = r'c:\Users\Ilya3\AlgoYES\Cryptanium_work_algo_opt\Final_table_main\grid_portfolio_performance.csv'

# Load trend strategy data
try:
    trend_df = pd.read_csv(trend_file, index_col=0, parse_dates=True)
    trend_df.index = pd.to_datetime(trend_df.index)
    print(f"Successfully loaded trend data with {len(trend_df)} rows")
    print(f"Date range: {trend_df.index.min()} to {trend_df.index.max()}")
    
    # Load grid strategy data
    grid_df = pd.read_csv(grid_file, index_col=0, parse_dates=True)
    grid_df.index = pd.to_datetime(grid_df.index)
    print(f"\nSuccessfully loaded grid data with {len(grid_df)} rows")
    print(f"Date range: {grid_df.index.min()} to {grid_df.index.max()}")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    raise

# Ensure we only use the overlapping date range
start_date = max(trend_df.index.min(), grid_df.index.min())
end_date = min(trend_df.index.max(), grid_df.index.max())

trend_df = trend_df[(trend_df.index >= start_date) & (trend_df.index <= end_date)]
grid_df = grid_df[(grid_df.index >= start_date) & (grid_df.index <= end_date)]

# Apply 6.5x leverage to all returns
LEVERAGE = 6.5
print(f"\nApplying {LEVERAGE}x leverage to all returns")

# Apply leverage to trend strategy returns
trend_cols = [col for col in trend_df.columns if 'Return' in col or 'rebalance' in col.lower() or 'weight' in col.lower()]
for col in trend_cols:
    if trend_df[col].dtype in ['float64', 'int64']:  # Only apply to numeric columns that are likely returns
        trend_df[col] = trend_df[col] * LEVERAGE

# Apply leverage to grid strategy returns
grid_cols = [col for col in grid_df.columns if 'Return' in col or 'return' in col.lower() or 'weight' in col.lower()]
for col in grid_cols:
    if grid_df[col].dtype in ['float64', 'int64']:  # Only apply to numeric columns that are likely returns
        grid_df[col] = grid_df[col] * LEVERAGE

# Print summary statistics
print("\nTrend strategy summary:")
print(trend_df.describe())
print("\nGrid strategy summary:")
print(grid_df.describe())

# Combine strategies
# 1. No rebalancing combination
no_rebalance_combined = pd.DataFrame({
    'Trend': trend_df['Equal Weight No Rebalance'],
    'Grid': grid_df['Equal_Weighted_Returns']
}, index=trend_df.index)

# Equal weight between trend and grid strategies
no_rebalance_combined['Combined'] = (no_rebalance_combined['Trend'] + no_rebalance_combined['Grid']) / 2

# 2. With rebalancing combination
rebalance_combined = pd.DataFrame({
    'Trend': trend_df['Equal Weight With Rebalance'],
    'Grid': grid_df['User_Weighted_Returns']
}, index=trend_df.index)

# Equal weight between trend and grid strategies
rebalance_combined['Combined'] = (rebalance_combined['Trend'] + rebalance_combined['Grid']) / 2

print("\nCombined strategies summary:")
print("No rebalance combined:")
print(no_rebalance_combined.describe())
print("\nWith rebalance combined:")
print(rebalance_combined.describe())

def calculate_cumulative_returns(returns_df, initial_capital=100):
    """Calculate cumulative returns from a DataFrame of returns"""
    return initial_capital * (1 + returns_df).cumprod()

# Calculate cumulative returns for all series with initial capital of 100
no_rebalance_cum = calculate_cumulative_returns(no_rebalance_combined)
rebalance_cum = calculate_cumulative_returns(rebalance_combined)

def plot_cumulative_returns(df, title, filename):
    plt.figure(figsize=(14, 8))
    
    # Calculate statistics
    total_return = (df.iloc[-1] / df.iloc[0] - 1) * 100  # in percent
    max_drawdown = (df / df.cummax() - 1).min() * 100  # in percent
    
    # Plot cumulative returns
    for col in df.columns:
        plt.plot(df.index, df[col], label=f"{col}: {total_return[col]:.2f}%")
    
    plt.title(f'{title}\nCumulative Returns (Initial Capital = 100)')
    plt.legend()
    plt.grid(True)
    plt.ylabel('Portfolio Value')
    plt.xticks(rotation=45)
    
    # Add text with max drawdown information
    stats_text = '\n'.join([
        f"{col}: "
        f"Return={total_return[col]:.2f}%, "
        f"MaxDD={max_drawdown[col]:.2f}%"
        for col in df.columns
    ])
    plt.figtext(0.15, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Generate plots
plot_cumulative_returns(no_rebalance_cum, 'No Rebalancing', 'no_rebalance_returns.png')
plot_cumulative_returns(rebalance_cum, 'With Rebalancing', 'rebalance_returns.png')

# Generate quantstats reports
# For no rebalancing
qs.extend_pandas()

# Get script directory for saving files
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save individual reports
no_rebalance_combined.to_csv(os.path.join(script_dir, 'no_rebalance_returns.csv'))
rebalance_combined.to_csv(os.path.join(script_dir, 'rebalance_returns.csv'))

# Save cumulative returns
no_rebalance_cum.to_csv(os.path.join(script_dir, 'no_rebalance_cumulative.csv'))
rebalance_cum.to_csv(os.path.join(script_dir, 'rebalance_cumulative.csv'))

# Save comparison plot
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
for col in no_rebalance_cum.columns:
    plt.plot(no_rebalance_cum.index, no_rebalance_cum[col], label=col)
plt.title('Cumulative Returns - No Rebalancing')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for col in rebalance_cum.columns:
    plt.plot(rebalance_cum.index, rebalance_cum[col], label=col)
plt.title('Cumulative Returns - With Rebalancing')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'combined_strategies_comparison.png'))
plt.close()

def generate_quantstats_report(returns, title, output_file, benchmark=None):
    """Generate quantstats report with error handling"""
    try:
        # Create a simple benchmark if not provided
        if benchmark is None:
            # Create a zero benchmark with the same index as returns
            if isinstance(returns, pd.DataFrame):
                benchmark_returns = pd.Series(0, index=returns.index, name='Benchmark')
            else:
                benchmark_returns = pd.Series(0, index=returns.index, name=returns.name + '_benchmark' if returns.name else 'Benchmark')
        else:
            benchmark_returns = benchmark
        
        # Generate the report
        qs.reports.html(
            returns=returns,
            benchmark=benchmark_returns,
            output=output_file,
            title=title,
            download_filename=os.path.basename(output_file)
        )
        print(f"Successfully generated report: {os.path.basename(output_file)}")
        return True
    except Exception as e:
        print(f"Error generating report {output_file}: {str(e)}")
        return False

# Generate reports for individual strategies
print("\nGenerating individual strategy reports...")
generate_quantstats_report(
    no_rebalance_combined['Combined'],
    title='No Rebalancing Strategy',
    output_file=os.path.join(script_dir, 'no_rebalance_combined_report.html')
)

generate_quantstats_report(
    rebalance_combined['Combined'],
    title='With Rebalancing Strategy',
    output_file=os.path.join(script_dir, 'rebalance_combined_report.html')
)

# Create a combined comparison report
combined_returns = pd.DataFrame({
    'No_Rebalance': no_rebalance_combined['Combined'],
    'With_Rebalance': rebalance_combined['Combined']
})

# Generate comparison reports
print("\nGenerating comparison reports...")
generate_quantstats_report(
    combined_returns,
    title='Strategy Comparison: Rebalance vs No Rebalance',
    output_file=os.path.join(script_dir, 'strategy_comparison_report.html')
)

# Generate a simple comparison without quantstats full report
try:
    comparison_df = pd.DataFrame({
        'No_Rebalance': no_rebalance_combined['Combined'],
        'With_Rebalance': rebalance_combined['Combined']
    })
    
    # Calculate cumulative returns
    cum_returns = (1 + comparison_df).cumprod()
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=col)
    
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(script_dir, 'cumulative_returns_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative returns plot to {plot_path}")
    
except Exception as e:
    print(f"Error generating cumulative returns plot: {str(e)}")

def calculate_metrics(returns_series, risk_free_rate=0.05, benchmark_returns=None):
    """Calculate various performance metrics for a returns series
    
    Args:
        returns_series: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.05 or 5%)
        benchmark_returns: Optional benchmark returns for calculating relative metrics
    """
    if len(returns_series) < 2:
        metrics = {
            'Annual Return (%)': np.nan,
            'Annual Volatility (%)': np.nan,
            'Sharpe Ratio': np.nan,
            'Sortino Ratio': np.nan,
            'Max Drawdown (%)': np.nan,
            'Calmar Ratio': np.nan,
            'Win Rate (%)': np.nan,
            'Profit Factor': np.nan,
            'Value at Risk 5%': np.nan,
            'CVaR 5%': np.nan,
            'Sharpe/CVaR Delta Ratio': np.nan
        }
        if benchmark_returns is not None:
            metrics['Sharpe/CVaR Delta Ratio'] = np.nan
        return metrics
    
    # Convert returns to numpy array for faster calculations
    returns = returns_series.values if hasattr(returns_series, 'values') else returns_series
    
    # Calculate basic metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (365.25 * 24 / len(returns)) - 1  # Assuming hourly data
    
    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(365.25 * 24)  # Annualized
    
    # Sharpe Ratio
    sharpe = (annual_return - risk_free_rate) / (volatility + 1e-10)  # Add small number to avoid division by zero
    
    # Sortino Ratio (only downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(365.25 * 24) if len(downside_returns) > 0 else 0
    sortino = (annual_return - risk_free_rate) / (downside_std + 1e-10)
    
    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    if not isinstance(cum_returns, pd.Series):
        cum_returns = pd.Series(cum_returns)
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    
    # Calmar Ratio (Return over Max Drawdown)
    calmar = annual_return / (-max_drawdown + 1e-10) if max_drawdown < 0 else np.nan
    
    # Win Rate
    win_rate = np.mean(returns > 0) * 100
    
    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = -np.sum(returns[returns < 0])
    profit_factor = gross_profit / (gross_loss + 1e-10)
    
    # Value at Risk (5%)
    var_5 = np.percentile(returns, 5) if len(returns) > 0 else np.nan
    
    # Expected Shortfall (5%)
    es_5 = np.mean(returns[returns <= var_5]) if (len(returns[returns <= var_5]) > 0) else np.nan
    
    # Calculate Ulcer Index
    if len(cum_returns) > 0:
        # Calculate drawdowns from peak
        peak = cum_returns.cummax()
        drawdowns = (peak - cum_returns) / peak
        # Calculate squared drawdowns and mean
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2)) * 100  # as percentage
    else:
        ulcer_index = np.nan
    
    # Calculate Ulcer Performance Index (Sharpe-like ratio using Ulcer Index)
    ulcer_performance_index = (annual_return * 100) / (ulcer_index + 1e-10) if not np.isnan(ulcer_index) and ulcer_index > 0 else np.nan
    
    # Calculate metrics dictionary
    metrics = {
        'Annual Return (%)': annual_return * 100,
        'Annual Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe,
        'Ulcer Index (%)': ulcer_index,
        'Ulcer Performance Index': ulcer_performance_index,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_drawdown * 100 if max_drawdown is not None else np.nan,
        'Calmar Ratio': calmar,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Value at Risk 5%': var_5 * 100 if var_5 is not None else np.nan,
        'CVaR 5%': es_5 * 100 if es_5 is not None and not np.isnan(es_5) else np.nan
    }
    
    # Calculate Sharpe/CVaR Delta Ratio if benchmark returns are provided
    if benchmark_returns is not None and len(benchmark_returns) == len(returns_series):
        # Calculate benchmark metrics
        benchmark_metrics = calculate_metrics(benchmark_returns, risk_free_rate)
        
        # Calculate deltas
        delta_sharpe = sharpe - benchmark_metrics['Sharpe Ratio']
        delta_cvar = metrics['CVaR 5%'] - benchmark_metrics['CVaR 5%']
        
        # Calculate Sharpe/CVaR Delta Ratio (avoid division by zero)
        if delta_cvar != 0:
            metrics['Sharpe/CVaR Delta Ratio'] = delta_sharpe / delta_cvar if not np.isnan(delta_sharpe) and not np.isnan(delta_cvar) else np.nan
        else:
            metrics['Sharpe/CVaR Delta Ratio'] = np.nan if delta_sharpe == 0 else np.sign(delta_sharpe) * np.inf
    
    return metrics

def optimize_weights(trend_returns, grid_returns, steps=20):
    """Optimize portfolio weights between trend and grid strategies
    
    Args:
        trend_returns: Series of returns from the trend strategy
        grid_returns: Series of returns from the grid strategy
        steps: Number of weight steps between 0% and 100%
        
    Returns:
        DataFrame with metrics for each weight combination and their deltas
    """
    results = []
    prev_metrics = None
    
    for grid_weight in np.linspace(0, 1, steps + 1):
        # Calculate combined returns
        combined_returns = grid_returns * grid_weight + trend_returns * (1 - grid_weight)
        
        # Calculate metrics
        metrics = calculate_metrics(combined_returns)
        metrics['Grid Weight'] = grid_weight * 100
        metrics['Trend Weight'] = (1 - grid_weight) * 100
        
        # Calculate deltas from previous weight if available
        if prev_metrics is not None:
            # Deltas for risk metrics
            metrics['Delta Sharpe'] = metrics['Sharpe Ratio'] - prev_metrics['Sharpe Ratio']
            metrics['Delta CVaR'] = metrics['CVaR 5%'] - prev_metrics['CVaR 5%']
            metrics['Delta Max Drawdown'] = metrics['Max Drawdown (%)'] - prev_metrics['Max Drawdown (%)']
            metrics['Delta Calmar'] = metrics['Calmar Ratio'] - prev_metrics['Calmar Ratio']
            
            # Calculate ratios
            if metrics['Delta CVaR'] != 0:
                metrics['Sharpe/CVaR Delta Ratio'] = metrics['Delta Sharpe'] / metrics['Delta CVaR']
            else:
                metrics['Sharpe/CVaR Delta Ratio'] = np.nan
                
            if metrics['Delta Max Drawdown'] != 0:
                metrics['Calmar/Drawdown Delta Ratio'] = metrics['Delta Calmar'] / metrics['Delta Max Drawdown']
            else:
                metrics['Calmar/Drawdown Delta Ratio'] = np.nan
        else:
            # Initialize deltas as NaN for the first row
            metrics['Delta Sharpe'] = np.nan
            metrics['Delta CVaR'] = np.nan
            metrics['Delta Max Drawdown'] = np.nan
            metrics['Delta Calmar'] = np.nan
            metrics['Sharpe/CVaR Delta Ratio'] = np.nan
            metrics['Calmar/Drawdown Delta Ratio'] = np.nan
        
        prev_metrics = metrics.copy()
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put weights and key metrics first
    col_order = [
        'Grid Weight', 'Trend Weight',
        'Annual Return (%)', 'Annual Volatility (%)',
        'Sharpe Ratio', 'Delta Sharpe',
        'Ulcer Index (%)', 'Ulcer Performance Index',
        'CVaR 5%', 'Delta CVaR',
        'Sharpe/CVaR Delta Ratio',
        'Max Drawdown (%)', 'Delta Max Drawdown',
        'Calmar Ratio', 'Delta Calmar',
        'Calmar/Drawdown Delta Ratio',
        'Sortino Ratio', 'Win Rate (%)',
        'Profit Factor', 'Value at Risk 5%'
    ]
    
    # Only include columns that exist in the results
    col_order = [col for col in col_order if col in results_df.columns]
    
    return results_df[col_order]

# Generate weight optimization report
def generate_weight_optimization_report(trend_returns, grid_returns, output_file):
    """Generate Excel report with weight optimization results"""
    # Calculate metrics for different weight combinations
    optimization_results = optimize_weights(trend_returns, grid_returns, steps=20)
    
    # Reorder columns for better readability
    columns_order = ['Grid Weight', 'Trend Weight'] + \
                   [col for col in optimization_results.columns if col not in ['Grid Weight', 'Trend Weight']]
    optimization_results = optimization_results[columns_order]
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Save optimization results
        optimization_results.to_excel(writer, sheet_name='Optimization Results', index=False)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Optimization Results']
        
        # Add a header format
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Format the header row
        for col_num, value in enumerate(optimization_results.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Format numbers
        percent_fmt = workbook.add_format({'num_format': '0.00%'})
        float_fmt = workbook.add_format({'num_format': '0.0000'})
        
        # Apply formats
        worksheet.set_column('A:B', 15)
        worksheet.set_column('C:L', 15, float_fmt)
        
        # Add a chart
        chart = workbook.add_chart({'type': 'line'})
        
        # Get the number of rows in the data
        max_row = len(optimization_results) + 1
        
        # Configure the chart
        chart.add_series({
            'name': 'Sharpe Ratio',
            'categories': ['Optimization Results', 1, 0, max_row, 0],
            'values': ['Optimization Results', 1, 3, max_row, 3],
            'y2_axis': True
        })
        
        chart.add_series({
            'name': 'Annual Return (%)',
            'categories': ['Optimization Results', 1, 0, max_row, 0],
            'values': ['Optimization Results', 1, 2, max_row, 2],
        })
        
        chart.set_title({'name': 'Strategy Optimization'})
        chart.set_x_axis({'name': 'Grid Weight (%)'})
        chart.set_y_axis({'name': 'Return (%)'})
        chart.set_y2_axis({'name': 'Sharpe Ratio'})
        
        # Insert the chart into the worksheet
        worksheet.insert_chart('N2', chart)
        
        # Add a summary sheet with best performing weights
        best_sharpe = optimization_results.nlargest(1, 'Sharpe Ratio').iloc[0]
        best_return = optimization_results.nlargest(1, 'Annual Return (%)').iloc[0]
        best_sortino = optimization_results.nlargest(1, 'Sortino Ratio').iloc[0]
        best_calmar = optimization_results.nlargest(1, 'Calmar Ratio').iloc[0]
        
        summary_data = {
            'Metric': ['Best by Sharpe', 'Best by Return', 'Best by Sortino', 'Best by Calmar'],
            'Grid Weight (%)': [
                best_sharpe['Grid Weight'],
                best_return['Grid Weight'],
                best_sortino['Grid Weight'],
                best_calmar['Grid Weight']
            ],
            'Trend Weight (%)': [
                best_sharpe['Trend Weight'],
                best_return['Trend Weight'],
                best_sortino['Trend Weight'],
                best_calmar['Trend Weight']
            ],
            'Sharpe Ratio': [
                best_sharpe['Sharpe Ratio'],
                best_return['Sharpe Ratio'],
                best_sortino['Sharpe Ratio'],
                best_calmar['Sharpe Ratio']
            ],
            'Annual Return (%)': [
                best_sharpe['Annual Return (%)'],
                best_return['Annual Return (%)'],
                best_sortino['Annual Return (%)'],
                best_calmar['Annual Return (%)']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Best Weights', index=False)
        
        # Format the summary sheet
        worksheet = writer.sheets['Best Weights']
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Format numbers in summary sheet
        worksheet.set_column('A:D', 15, float_fmt)
        
        # Add a chart of efficient frontier
        chart = workbook.add_chart({'type': 'scatter'})
        
        chart.add_series({
            'name': 'Efficient Frontier',
            'categories': ['Optimization Results', 1, 2, max_row, 2],
            'values': ['Optimization Results', 1, 1, max_row, 1],
            'marker': {'type': 'circle', 'size': 6},
            'trendline': {'type': 'polynomial', 'order': 2, 'display_equation': False},
        })
        
        chart.set_title({'name': 'Efficient Frontier'})
        chart.set_x_axis({'name': 'Risk (Volatility %)'})
        chart.set_y_axis({'name': 'Return (%)'})
        
        # Insert the chart into the worksheet
        worksheet.insert_chart('F2', chart)

# Generate weight optimization reports
print("\nGenerating weight optimization reports...")

# For no rebalancing strategy
output_file_no_rebalance = os.path.join(script_dir, 'strategy_optimization_no_rebalance.xlsx')
generate_weight_optimization_report(
    no_rebalance_combined['Trend'],
    no_rebalance_combined['Grid'],
    output_file_no_rebalance
)

# For rebalancing strategy
output_file_rebalance = os.path.join(script_dir, 'strategy_optimization_rebalance.xlsx')
generate_weight_optimization_report(
    rebalance_combined['Trend'],
    rebalance_combined['Grid'],
    output_file_rebalance
)

print(f"\nAnalysis complete. Files saved in: {script_dir}")
print("\nHTML Reports:")
print("- no_rebalance_combined_report.html")
print("- rebalance_combined_report.html")
print("- strategy_comparison_report.html")
print("- strategy_tearsheet.html")
print("- full_comparison_report.html")

print("\nData Files:")
print("- no_rebalance_returns.csv")
print("- rebalance_returns.csv")
print("- no_rebalance_cumulative.csv")
print("- rebalance_cumulative.csv")
print("- combined_strategies_comparison.png")

print("\nOptimization Reports:")
print("- strategy_optimization_no_rebalance.xlsx (без ребалансировки)")
print("- strategy_optimization_rebalance.xlsx (с ребалансировкой)")
