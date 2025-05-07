"""
Visualization utilities for Search and Rescue experiments.
Provides functions for analyzing and visualizing experimental results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_experiment_data(file_paths):
    """
    Load and combine experiment data from multiple CSV files
    
    Args:
        file_paths: List of CSV file paths to load
        
    Returns:
        pd.DataFrame: Combined dataframe with all experiment data
    """
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def generate_heatmap(data, x_var, y_var, value_var, title, output_path):
    """
    Generate a heatmap visualization
    
    Args:
        data: DataFrame containing the data
        x_var: Column name for x-axis
        y_var: Column name for y-axis
        value_var: Column name for cell values
        title: Plot title
        output_path: Path to save the figure
    """
    # Create pivot table
    pivot = data.pivot_table(
        index=y_var, 
        columns=x_var, 
        values=value_var,
        aggfunc='mean'
    )
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_box_plots(data, x_var, y_var, hue_var, title, output_path):
    """
    Create box plots for comparing distributions
    
    Args:
        data: DataFrame containing the data
        x_var: Column name for x-axis categories
        y_var: Column name for y-axis values
        hue_var: Column name for color grouping
        title: Plot title
        output_path: Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=x_var, y=y_var, hue=hue_var, data=data)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_rescue_rates(data, group_vars, output_path):
    """
    Plot average rescue rates over time for different configurations
    
    Args:
        data: DataFrame with rescue_rate lists and grouping variables
        group_vars: List of columns to group by
        output_path: Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    # Group data
    grouped = data.groupby(group_vars)
    
    # Plot each group
    for name, group in grouped:
        # Extract rescue rates (stored as strings) and convert to lists
        rates = [eval(rate) for rate in group['rescue_rate'] if isinstance(rate, str)]
        
        if not rates:
            continue
            
        # Find the maximum length
        max_len = max(len(rate) for rate in rates)
        
        # Pad shorter arrays with their final value
        padded_rates = []
        for rate in rates:
            padded = rate.copy()
            if len(padded) < max_len:
                padded.extend([padded[-1]] * (max_len - len(padded)))
            padded_rates.append(padded)
        
        # Calculate average rescue rate at each time step
        avg_rate = np.mean(padded_rates, axis=0)
        
        # Create label based on group variables
        if isinstance(name, tuple):
            label = " - ".join([str(n) for n in name])
        else:
            label = str(name)
            
        # Plot
        plt.plot(avg_rate, label=label)
    
    plt.title('Average Rescue Rate Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Fraction of Victims Rescued')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def perform_anova(data, value_col, group_cols):
    """
    Perform one-way ANOVA for each grouping variable
    
    Args:
        data: DataFrame with the data
        value_col: Column to analyze
        group_cols: List of columns to group by for ANOVA tests
    
    Returns:
        dict: ANOVA results
    """
    results = {}
    
    for col in group_cols:
        # Get unique groups
        groups = data[col].unique()
        
        # Extract data for each group
        group_data = [data[data[col] == group][value_col].dropna() for group in groups]
        
        # Run ANOVA
        f_val, p_val = stats.f_oneway(*group_data)
        
        results[col] = {
            'F-value': f_val,
            'p-value': p_val,
            'significant': p_val < 0.05
        }
    
    return results

def generate_interaction_plots(data, x_var, trace_var, response_var, output_path):
    """
    Generate interaction plots to show how two factors affect a response variable
    
    Args:
        data: DataFrame with the data
        x_var: Variable for the x-axis
        trace_var: Variable for different traces
        response_var: Response variable
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Get unique values for the trace variable
    trace_values = data[trace_var].unique()
    
    # Create line for each trace value
    for trace_val in trace_values:
        subset = data[data[trace_var] == trace_val]
        
        # Calculate mean response for each x value
        means = subset.groupby(x_var)[response_var].mean()
        
        plt.plot(means.index, means.values, marker='o', label=str(trace_val))
    
    plt.title(f'Interaction Plot: {x_var} x {trace_var} on {response_var}')
    plt.xlabel(x_var)
    plt.ylabel(response_var)
    plt.legend(title=trace_var)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def summarize_results(data, output_path):
    """
    Generate a statistical summary of results
    
    Args:
        data: DataFrame with experiment results
        output_path: Path to save the summary text file
    """
    with open(output_path, 'w') as f:
        f.write("SEARCH AND RESCUE EXPERIMENT SUMMARY\n")
        f.write("===================================\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"Total trials: {len(data)}\n")
        f.write(f"Average completion time: {data['Completion_Time'].mean():.2f} steps\n")
        f.write(f"Average coverage: {data['Coverage'].mean()*100:.2f}%\n")
        f.write(f"Average success rate: {data['Success_Rate'].mean()*100:.2f}%\n\n")
        
        # Strategy comparison
        f.write("STRATEGY COMPARISON\n")
        f.write("------------------\n")
        strategy_stats = data.groupby('Strategy').agg({
            'Completion_Time': ['mean', 'std'],
            'Coverage': ['mean', 'std'],
            'Success_Rate': ['mean', 'std'],
            'Avg_Path_Length': ['mean', 'std'],
            'Avg_Idle_Time': ['mean', 'std']
        })
        
        for strategy in strategy_stats.index:
            f.write(f"Strategy: {strategy}\n")
            f.write(f"  Completion Time: {strategy_stats.loc[strategy, ('Completion_Time', 'mean')]:.2f} ± {strategy_stats.loc[strategy, ('Completion_Time', 'std')]:.2f}\n")
            f.write(f"  Coverage: {strategy_stats.loc[strategy, ('Coverage', 'mean')]*100:.2f}% ± {strategy_stats.loc[strategy, ('Coverage', 'std')]*100:.2f}%\n")
            f.write(f"  Success Rate: {strategy_stats.loc[strategy, ('Success_Rate', 'mean')]*100:.2f}% ± {strategy_stats.loc[strategy, ('Success_Rate', 'std')]*100:.2f}%\n")
            f.write(f"  Avg Path Length: {strategy_stats.loc[strategy, ('Avg_Path_Length', 'mean')]:.2f} ± {strategy_stats.loc[strategy, ('Avg_Path_Length', 'std')]:.2f}\n")
            f.write(f"  Avg Idle Time: {strategy_stats.loc[strategy, ('Avg_Idle_Time', 'mean')]:.2f} ± {strategy_stats.loc[strategy, ('Avg_Idle_Time', 'std')]:.2f}\n\n")
        
        # Agent count comparison
        f.write("AGENT COUNT COMPARISON\n")
        f.write("---------------------\n")
        count_stats = data.groupby('Agent_Count').agg({
            'Completion_Time': ['mean', 'std'],
            'Coverage': ['mean', 'std'],
            'Success_Rate': ['mean', 'std']
        })
        
        for count in count_stats.index:
            f.write(f"Agent Count: {count}\n")
            f.write(f"  Completion Time: {count_stats.loc[count, ('Completion_Time', 'mean')]:.2f} ± {count_stats.loc[count, ('Completion_Time', 'std')]:.2f}\n")
            f.write(f"  Coverage: {count_stats.loc[count, ('Coverage', 'mean')]*100:.2f}% ± {count_stats.loc[count, ('Coverage', 'std')]*100:.2f}%\n")
            f.write(f"  Success Rate: {count_stats.loc[count, ('Success_Rate', 'mean')]*100:.2f}% ± {count_stats.loc[count, ('Success_Rate', 'std')]*100:.2f}%\n\n")
        
        # Statistical significance tests
        f.write("STATISTICAL SIGNIFICANCE\n")
        f.write("------------------------\n")
        
        # ANOVA for strategy effect
        strategies = data['Strategy'].unique()
        strategy_groups = [data[data['Strategy'] == s]['Completion_Time'].dropna() for s in strategies]
        
        if all(len(g) > 0 for g in strategy_groups):
            f_val, p_val = stats.f_oneway(*strategy_groups)
            f.write(f"Strategy effect on Completion Time: F={f_val:.2f}, p={p_val:.4f}")
            if p_val < 0.05:
                f.write(" (significant)\n")
            else:
                f.write(" (not significant)\n")
        
        # ANOVA for agent count effect
        counts = data['Agent_Count'].unique()
        count_groups = [data[data['Agent_Count'] == c]['Completion_Time'].dropna() for c in counts]
        
        if all(len(g) > 0 for g in count_groups):
            f_val, p_val = stats.f_oneway(*count_groups)
            f.write(f"Agent Count effect on Completion Time: F={f_val:.2f}, p={p_val:.4f}")
            if p_val < 0.05:
                f.write(" (significant)\n")
            else:
                f.write(" (not significant)\n")
        
        f.write("\nNote: A p-value less than 0.05 indicates a statistically significant effect.\n")

def create_comprehensive_report(data_files, output_dir="experiment_analysis"):
    """
    Create a comprehensive analysis of experiment results with visualizations
    
    Args:
        data_files: List of CSV data files
        output_dir: Directory to save the analysis
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data
    data = load_experiment_data(data_files)
    
    # Generate statistical summary
    summarize_results(data, f"{output_dir}/summary_statistics.txt")
    
    # Create standard visualizations
    
    # 1. Completion Time by Strategy and Agent Count
    generate_heatmap(
        data, 
        'Agent_Count', 
        'Strategy', 
        'Completion_Time',
        'Average Completion Time',
        f"{output_dir}/completion_time_heatmap.png"
    )
    
    # 2. Coverage by Strategy and Agent Count
    generate_heatmap(
        data, 
        'Agent_Count', 
        'Strategy', 
        'Coverage',
        'Average Coverage (%)',
        f"{output_dir}/coverage_heatmap.png"
    )
    
    # 3. Success Rate by Strategy and Agent Count
    generate_heatmap(
        data, 
        'Agent_Count', 
        'Strategy', 
        'Success_Rate',
        'Average Success Rate (%)',
        f"{output_dir}/success_rate_heatmap.png"
    )
    
    # 4. Box plots for completion time by strategy
    create_box_plots(
        data,
        'Strategy',
        'Completion_Time',
        'Agent_Count',
        'Completion Time Distribution by Strategy',
        f"{output_dir}/completion_time_boxplot.png"
    )
    
    # 5. Box plots for coverage by strategy
    create_box_plots(
        data,
        'Strategy',
        'Coverage',
        'Agent_Count',
        'Coverage Distribution by Strategy',
        f"{output_dir}/coverage_boxplot.png"
    )
    
    # 6. Interaction plots
    generate_interaction_plots(
        data,
        'Agent_Count',
        'Strategy',
        'Completion_Time',
        f"{output_dir}/interaction_completion_time.png"
    )
    
    generate_interaction_plots(
        data,
        'Agent_Count',
        'Strategy',
        'Coverage',
        f"{output_dir}/interaction_coverage.png"
    )
    
    # Phase-specific visualizations
    if 'Phase' in data.columns:
        # Separate data by phase
        phase1_data = data[data['Phase'] == 1]
        phase2_data = data[data['Phase'] == 2]
        phase3_data = data[data['Phase'] == 3]
        
        # Phase 2: Environmental complexity analysis
        if not phase2_data.empty and 'Obstacle_Density' in phase2_data.columns:
            # Create visualizations for environmental impact
            create_box_plots(
                phase2_data,
                'Obstacle_Density',
                'Completion_Time',
                'Strategy',
                'Impact of Environmental Complexity on Completion Time',
                f"{output_dir}/environment_completion_time.png"
            )
            
            generate_interaction_plots(
                phase2_data,
                'Obstacle_Density',
                'Strategy',
                'Coverage',
                f"{output_dir}/environment_coverage_interaction.png"
            )
        
        # Phase 3: Communication constraints analysis
        if not phase3_data.empty and 'Comm_Range' in phase3_data.columns:
            # Create visualizations for communication impact
            create_box_plots(
                phase3_data,
                'Comm_Range',
                'Completion_Time',
                'Agent_Count',
                'Impact of Communication Range on Completion Time',
                f"{output_dir}/communication_completion_time.png"
            )
            
            generate_interaction_plots(
                phase3_data,
                'Comm_Range',
                'Agent_Count',
                'Success_Rate',
                f"{output_dir}/communication_success_interaction.png"
            )
    
    print(f"Comprehensive analysis created in {output_dir}")
    return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Search and Rescue experiment results')
    parser.add_argument('files', nargs='+', help='CSV files with experiment results')
    parser.add_argument('--output', default='experiment_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    create_comprehensive_report(args.files, args.output)