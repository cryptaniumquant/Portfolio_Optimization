import os
import pandas as pd
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime

def calculate_returns(df, use_rebalancing=False):
    """
    Calculate returns for a strategy with or without rebalancing
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with strategy data including 'backtest_Value', 'real_Value', 'rebalance_point', and 'weight'
    use_rebalancing : bool
        If True, apply rebalancing based on weight column
        If False, simply hold the strategy
        
    Returns:
    --------
    pandas.Series
        Daily returns series
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Объединяем столбцы backtest_Value и real_Value в один
    # Если значение в real_Value не NaN, используем его, иначе используем backtest_Value
    df_copy['value'] = df_copy['real_Value'].fillna(df_copy['backtest_Value'])
    
    # Проверяем минимальные и максимальные значения
    print(f"Min value: {df_copy['value'].min()}, Max value: {df_copy['value'].max()}")
    
    # Рассчитываем NAV по формуле NAV = 1 + (PnL - PnL[0])/100
    if use_rebalancing:
        # С ребалансировкой - учитываем веса
        adjusted_pnl = []
        current_pnl = 0
        
        for i in range(len(df_copy)):
            if i > 0:
                # Если вес равен 0, не меняем PnL
                if df_copy['weight'].iloc[i-1] > 0:
                    pnl_change = df_copy['value'].iloc[i] - df_copy['value'].iloc[i-1]
                    current_pnl += pnl_change
            
            adjusted_pnl.append(current_pnl)
        
        # Преобразуем в Series
        adjusted_pnl_series = pd.Series(adjusted_pnl, index=df_copy.index)
        
        # Рассчитываем NAV по формуле NAV = 1 + (PnL - PnL[0])/100
        nav = 1 + (adjusted_pnl_series - adjusted_pnl_series.iloc[0]) / 100
    else:
        # Без ребалансировки - просто используем значения из файла
        nav = 1 + (df_copy['value'] - df_copy['value'].iloc[0]) / 100
    
    # Рассчитываем доходность на основе NAV
    returns = nav.pct_change().fillna(0)
    
    # Ресемплируем к дневной частоте для QuantStats (если данные имеют более высокую частоту)
    if returns.index.inferred_freq != '1h':
        returns = returns.resample('1h').max().fillna(0)
    
    return returns

def analyze_strategy(file_path, output_dir='reports', returns_data=None):
    """
    Analyze a strategy with and without rebalancing using QuantStats
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file with strategy data
    output_dir : str
        Directory to save the reports
    returns_data : dict, optional
        Dictionary to store returns data for combined analysis
    
    Returns:
    --------
    pandas.DataFrame
        Comparison DataFrame with metrics for both approaches
    """
    # Extract strategy name from file path
    file_name = os.path.basename(file_path)
    strategy_name = file_name.replace('merged_', '').replace('_joined_with_peak.csv', '')
    
    print(f"\nAnalyzing strategy: {strategy_name}")
    
    # Load the data
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Skip if no backtest_Value column
    if 'backtest_Value' not in df.columns:
        print(f"No 'backtest_Value' column in {file_path}, skipping.")
        return
    
    # Calculate returns for both approaches
    returns_no_rebalance = calculate_returns(df, use_rebalancing=False)
    returns_with_rebalance = calculate_returns(df, use_rebalancing=True)
    
    # Store returns data for combined analysis if dictionary is provided
    if returns_data is not None:
        returns_data[f"{strategy_name}_no_rebalance"] = returns_no_rebalance
        returns_data[f"{strategy_name}_with_rebalance"] = returns_with_rebalance
    
    # Create output directory if it doesn't exist
    # Убедимся, что путь использует правильные разделители
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Пропускаем генерацию отдельных отчетов для экономии времени
    
    # Compare the two approaches
    print("\nComparison of key metrics:")
    metrics = [
        ('Cumulative Return', lambda r: qs.stats.comp(r)),
        ('CAGR', lambda r: qs.stats.cagr(r)),
        ('Sharpe Ratio', lambda r: qs.stats.sharpe(r)),
        ('Max Drawdown', lambda r: qs.stats.max_drawdown(r)),
        ('Win Rate', lambda r: qs.stats.win_rate(r)),
        ('Volatility (Ann.)', lambda r: qs.stats.volatility(r, annualize=True)),
        ('Sortino Ratio', lambda r: qs.stats.sortino(r)),
        ('Calmar Ratio', lambda r: qs.stats.calmar(r))
    ]
    
    comparison = pd.DataFrame(index=[m[0] for m in metrics], columns=['No Rebalance', 'With Rebalance'])
    
    for metric_name, metric_func in metrics:
        try:
            comparison.loc[metric_name, 'No Rebalance'] = metric_func(returns_no_rebalance)
            comparison.loc[metric_name, 'With Rebalance'] = metric_func(returns_with_rebalance)
        except Exception as e:
            comparison.loc[metric_name, 'No Rebalance'] = np.nan
            comparison.loc[metric_name, 'With Rebalance'] = np.nan
            print(f"Error calculating {metric_name}: {e}")
    
    print(comparison)
    
    # Generate comparison report
    print("\nGenerating comparison report...")
    report_path = os.path.join(output_dir, f"{strategy_name}_comparison.html")
    qs.reports.html(
        returns_with_rebalance, 
        benchmark=returns_no_rebalance,
        output=report_path,
        title=f"{strategy_name} - Rebalance vs No Rebalance",
        benchmark_title="BACKTEST_VALUE"  # Явно указываем название бенчмарка
    )
    
    return comparison

def create_combined_equal_weight_report(returns_data, output_dir):
    """
    Create a combined report with two graphs:
    1. All strategies with equal weight without rebalancing
    2. All strategies with equal weight with rebalancing
    
    Parameters:
    -----------
    returns_data : dict
        Dictionary containing returns data for all strategies
    output_dir : str
        Directory to save the report
    """
    print("\nCreating combined equal weight report...")
    
    # Separate no_rebalance and with_rebalance returns
    no_rebalance_returns = {k: v for k, v in returns_data.items() if 'no_rebalance' in k}
    with_rebalance_returns = {k: v for k, v in returns_data.items() if 'with_rebalance' in k}
    
    # Create equal weight portfolios
    if not no_rebalance_returns or not with_rebalance_returns:
        print("Not enough data to create combined report.")
        return
    
    # Align all returns to the same timeframe
    # For no_rebalance portfolio
    no_rebal_dfs = list(no_rebalance_returns.values())
    common_index = no_rebal_dfs[0].index
    for df in no_rebal_dfs[1:]:
        common_index = common_index.intersection(df.index)
    
    # Create equal weight portfolio for no_rebalance
    no_rebal_aligned = pd.DataFrame({k: v.reindex(common_index) for k, v in no_rebalance_returns.items()})
    equal_weight_no_rebal = no_rebal_aligned.mean(axis=1)  # Equal weight
    
    # For with_rebalance portfolio
    with_rebal_dfs = list(with_rebalance_returns.values())
    common_index = with_rebal_dfs[0].index
    for df in with_rebal_dfs[1:]:
        common_index = common_index.intersection(df.index)
    
    # Create equal weight portfolio for with_rebalance
    with_rebal_aligned = pd.DataFrame({k: v.reindex(common_index) for k, v in with_rebalance_returns.items()})
    equal_weight_with_rebal = with_rebal_aligned.mean(axis=1)  # Equal weight
    
    # Save returns data to CSV for future use
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    returns_dir = os.path.join(output_dir, "returns_data")
    os.makedirs(returns_dir, exist_ok=True)
    
    # Save individual strategy returns
    for strategy, returns in returns_data.items():
        returns.to_csv(os.path.join(returns_dir, f"{strategy}_returns.csv"))
    
    # Save equal weight portfolios
    equal_weight_no_rebal.to_csv(os.path.join(returns_dir, "equal_weight_no_rebalance_returns.csv"))
    equal_weight_with_rebal.to_csv(os.path.join(returns_dir, "equal_weight_with_rebalance_returns.csv"))
    
    # Create combined DataFrame for visualization
    combined_returns = pd.DataFrame({
        'Equal Weight No Rebalance': equal_weight_no_rebal,
        'Equal Weight With Rebalance': equal_weight_with_rebal
    })
    combined_returns.to_csv(os.path.join(returns_dir, "combined_equal_weight_returns.csv"))
    print(f"Saved returns data to {returns_dir}")
    
    # Create the combined report
    report_path = os.path.join(output_dir, "equal_weight_portfolios_comparison.html")
    qs.reports.html(
        equal_weight_with_rebal,
        benchmark=equal_weight_no_rebal,
        output=report_path,
        title="Equal Weight Portfolios - Rebalance vs No Rebalance",
        benchmark_title="Equal Weight No Rebalance"
    )
    print(f"Saved combined report to {report_path}")
    
    # Create a custom visualization with two graphs
    plt.figure(figsize=(12, 12))
    
    # Calculate cumulative returns for plotting
    cum_returns_no_rebal = (1 + equal_weight_no_rebal).cumprod()
    cum_returns_with_rebal = (1 + equal_weight_with_rebal).cumprod()
    
    # Plot 1: Equal weight strategies without rebalancing
    plt.subplot(2, 1, 1)
    for strategy, returns in no_rebalance_returns.items():
        strategy_name = strategy.replace('_no_rebalance', '')
        cum_returns = (1 + returns.reindex(common_index)).cumprod()
        plt.plot(cum_returns.index, cum_returns.values, alpha=0.5, linewidth=1, label=strategy_name)
    
    # Add the equal weight portfolio with thicker line
    plt.plot(cum_returns_no_rebal.index, cum_returns_no_rebal.values, 'k-', linewidth=2, label='Equal Weight Portfolio')
    
    plt.title('Equal Weight Strategies Without Rebalancing')
    plt.ylabel('Cumulative Returns')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize='small')
    
    # Plot 2: Equal weight strategies with rebalancing
    plt.subplot(2, 1, 2)
    for strategy, returns in with_rebalance_returns.items():
        strategy_name = strategy.replace('_with_rebalance', '')
        cum_returns = (1 + returns.reindex(common_index)).cumprod()
        plt.plot(cum_returns.index, cum_returns.values, alpha=0.5, linewidth=1, label=strategy_name)
    
    # Add the equal weight portfolio with thicker line
    plt.plot(cum_returns_with_rebal.index, cum_returns_with_rebal.values, 'k-', linewidth=2, label='Equal Weight Portfolio')
    
    plt.title('Equal Weight Strategies With Rebalancing')
    plt.ylabel('Cumulative Returns')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize='small')
    
    plt.tight_layout()
    
    # Save the figure
    chart_path = os.path.join(output_dir, "equal_weight_strategies_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved equal weight strategies comparison chart to {chart_path}")

def main():
    # Get all strategy CSV files
    data_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob(os.path.join(data_dir, '*_joined_with_peak.csv'))
    
    if not csv_files:
        print("No strategy files found with '_joined_with_peak.csv' suffix.")
        return
    
    print(f"Found {len(csv_files)} strategy files to analyze.")
    
    # Create output directory
    output_dir = os.path.join(data_dir, "reports", "quantstats")
    # Убедимся, что путь использует правильные разделители
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store returns data for combined analysis
    returns_data = {}
    
    # Analyze each strategy
    all_comparisons = {}
    for file_path in csv_files:
        comparison = analyze_strategy(file_path, output_dir, returns_data)
        if comparison is not None:
            strategy_name = os.path.basename(file_path).replace('merged_', '').replace('_joined_with_peak.csv', '')
            all_comparisons[strategy_name] = comparison
    
    # Create combined equal weight report
    if returns_data:
        create_combined_equal_weight_report(returns_data, output_dir)
    
    # Create summary of all strategies
    if all_comparisons:
        print("\nCreating summary of all strategies...")
        # For each metric, create a summary table
        metrics = all_comparisons[list(all_comparisons.keys())[0]].index
        
        # Create a directory for summary reports
        summary_dir = os.path.join(output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Create a summary DataFrame for all metrics
        all_metrics_summary = pd.DataFrame()
        
        for metric in metrics:
            summary = pd.DataFrame(columns=['No Rebalance', 'With Rebalance', 'Difference', '% Improvement'])
            
            for strategy, comparison in all_comparisons.items():
                no_rebal = comparison.loc[metric, 'No Rebalance']
                with_rebal = comparison.loc[metric, 'With Rebalance']
                
                if not (pd.isna(no_rebal) or pd.isna(with_rebal)):
                    diff = with_rebal - no_rebal
                    pct_improvement = diff / abs(no_rebal) * 100 if no_rebal != 0 else np.nan
                    
                    summary.loc[strategy] = [no_rebal, with_rebal, diff, pct_improvement]
            
            # Выводим сводку в консоль
            print(f"\nSummary for {metric}:")
            print(summary)
            
            # Сохраняем сводку в CSV
            summary_file = os.path.join(summary_dir, f"{metric.replace(' ', '_').lower()}_summary.csv")
            summary.to_csv(summary_file)
            print(f"Saved summary to {summary_file}")
            
            # Add to all metrics summary
            all_metrics_summary[f"{metric}_No_Rebalance"] = summary['No Rebalance']
            all_metrics_summary[f"{metric}_With_Rebalance"] = summary['With Rebalance']
            all_metrics_summary[f"{metric}_Improvement_%"] = summary['% Improvement']
        
        # Save the complete summary
        complete_summary_file = os.path.join(summary_dir, "all_metrics_summary.csv")
        all_metrics_summary.to_csv(complete_summary_file)
        print(f"\nSaved complete metrics summary to {complete_summary_file}")
        
        # Create a visual comparison of strategies
        try:
            # Prepare data for visualization
            performance_comparison = pd.DataFrame(index=all_comparisons.keys())
            performance_comparison['CAGR_No_Rebalance'] = [comp.loc['CAGR', 'No Rebalance'] for comp in all_comparisons.values()]
            performance_comparison['CAGR_With_Rebalance'] = [comp.loc['CAGR', 'With Rebalance'] for comp in all_comparisons.values()]
            performance_comparison['Sharpe_No_Rebalance'] = [comp.loc['Sharpe Ratio', 'No Rebalance'] for comp in all_comparisons.values()]
            performance_comparison['Sharpe_With_Rebalance'] = [comp.loc['Sharpe Ratio', 'With Rebalance'] for comp in all_comparisons.values()]
            
            # Sort by CAGR with rebalancing
            performance_comparison = performance_comparison.sort_values('CAGR_With_Rebalance', ascending=False)
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            
            # Plot CAGR comparison
            plt.subplot(2, 1, 1)
            performance_comparison[['CAGR_No_Rebalance', 'CAGR_With_Rebalance']].plot(kind='bar', ax=plt.gca())
            plt.title('CAGR Comparison Across Strategies')
            plt.ylabel('CAGR')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot Sharpe comparison
            plt.subplot(2, 1, 2)
            performance_comparison[['Sharpe_No_Rebalance', 'Sharpe_With_Rebalance']].plot(kind='bar', ax=plt.gca())
            plt.title('Sharpe Ratio Comparison Across Strategies')
            plt.ylabel('Sharpe Ratio')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save the figure
            comparison_chart = os.path.join(summary_dir, "strategy_performance_comparison.png")
            plt.savefig(comparison_chart)
            plt.close()
            print(f"Saved strategy performance comparison chart to {comparison_chart}")
            
            # Create a heatmap of improvement percentages
            improvement_data = pd.DataFrame(index=all_comparisons.keys())
            for metric in metrics:
                metric_key = f"{metric}_Improvement_%"
                if metric_key in all_metrics_summary.columns:
                    improvement_data[metric] = all_metrics_summary[metric_key]
            
            # Handle NaN values
            improvement_data = improvement_data.fillna(0)
            
            # Create heatmap
            plt.figure(figsize=(12, len(improvement_data) * 0.5 + 2))
            plt.title('Improvement Percentage with Rebalancing')
            
            # Use a diverging colormap centered at 0
            cmap = plt.cm.RdYlGn  # Red for negative, green for positive
            norm = plt.Normalize(vmin=-50, vmax=50)  # Adjust range as needed
            
            plt.imshow(improvement_data.values, cmap=cmap, norm=norm, aspect='auto')
            plt.colorbar(label='% Improvement')
            
            # Add labels
            plt.yticks(range(len(improvement_data)), improvement_data.index)
            plt.xticks(range(len(improvement_data.columns)), improvement_data.columns, rotation=45, ha='right')
            
            # Add text annotations
            for i in range(len(improvement_data)):
                for j in range(len(improvement_data.columns)):
                    value = improvement_data.iloc[i, j]
                    color = 'black' if -20 < value < 20 else 'white'
                    plt.text(j, i, f"{value:.1f}%", ha='center', va='center', color=color)
            
            plt.tight_layout()
            
            # Save the heatmap
            heatmap_file = os.path.join(summary_dir, "improvement_heatmap.png")
            plt.savefig(heatmap_file)
            plt.close()
            print(f"Saved improvement heatmap to {heatmap_file}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    main()
