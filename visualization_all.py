import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Define file paths
phase1_path = 'experiment_results/phase1_results_20250507_092612.csv'
phase2_path = 'experiment_results/phase2_results_20250508_112037.csv'
phase3_path = 'experiment_results/phase3_results_20250508_112039.csv'

# Create output directory
output_dir = 'experiment_analysis/comprehensive'
os.makedirs(output_dir, exist_ok=True)

# Custom color palette for consistent visualization
colors = ['#2c7bb6', '#7fcdbb', '#d7191c']
sns.set_palette(sns.color_palette(colors))

# Helper function to ensure directory exists
def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

# Helper function for creating heatmaps
def create_heatmap(data, x_col, y_col, value_col, title, filename, fmt='.2f'):
    plt.figure(figsize=(10, 8))
    pivot = data.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='mean')
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=fmt, linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Helper function for creating box plots
def create_boxplot(data, x_col, y_col, hue_col, title, filename):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x_col, y=y_col, hue=hue_col, data=data)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Helper function for creating line plots
def create_lineplot(data, x_col, y_col, hue_col, title, filename, sort_key=None):
    plt.figure(figsize=(12, 6))
    
    # Group and calculate means
    grouped = data.groupby([hue_col, x_col])[y_col].mean().reset_index()
    
    # Plot each group
    for name, group_data in grouped.groupby(hue_col):
        # Sort if a sort key is provided
        if sort_key:
            group_data = group_data.sort_values(by=x_col, key=sort_key)
        plt.plot(group_data[x_col], group_data[y_col], marker='o', label=str(name))
    
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(f'Average {y_col}')
    plt.legend(title=hue_col)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# PART 1: ANALYZE PHASE 1 (AGENT COUNT AND STRATEGY COMPARISON)
print('Analyzing Phase 1: Agent Count and Strategy Comparison...')
phase1_dir = os.path.join(output_dir, 'phase1')
ensure_dir(phase1_dir)

try:
    # Load Phase 1 data
    phase1_data = pd.read_csv(phase1_path)
    
    # Create heatmaps
    create_heatmap(
        phase1_data, 'Agent_Count', 'Strategy', 'Completion_Time',
        'Average Completion Time by Strategy and Agent Count',
        os.path.join('phase1', 'completion_time_heatmap.png')
    )
    
    create_heatmap(
        phase1_data, 'Agent_Count', 'Strategy', 'Coverage',
        'Average Coverage by Strategy and Agent Count',
        os.path.join('phase1', 'coverage_heatmap.png')
    )
    
    # Create boxplots
    create_boxplot(
        phase1_data, 'Strategy', 'Completion_Time', 'Agent_Count',
        'Completion Time by Strategy and Agent Count',
        os.path.join('phase1', 'completion_time_boxplot.png')
    )
    
    create_boxplot(
        phase1_data, 'Strategy', 'Coverage', 'Agent_Count',
        'Coverage by Strategy and Agent Count',
        os.path.join('phase1', 'coverage_boxplot.png')
    )
    
    # Create interaction plots
    create_lineplot(
        phase1_data, 'Agent_Count', 'Completion_Time', 'Strategy',
        'Strategy Performance by Agent Count',
        os.path.join('phase1', 'strategy_by_agent_count.png')
    )
    
    create_lineplot(
        phase1_data, 'Strategy', 'Completion_Time', 'Agent_Count',
        'Agent Count Performance by Strategy',
        os.path.join('phase1', 'agent_count_by_strategy.png')
    )
    
    # Generate summary statistics
    with open(os.path.join(phase1_dir, 'summary_statistics.txt'), 'w') as f:
        f.write('PHASE 1: AGENT COUNT AND STRATEGY COMPARISON\n')
        f.write('===========================================\n\n')
        
        # Overall statistics
        f.write('OVERALL STATISTICS\n')
        f.write('-----------------\n')
        f.write(f'Total trials: {len(phase1_data)}\n')
        f.write(f'Average completion time: {phase1_data["Completion_Time"].mean():.2f} steps\n')
        f.write(f'Average coverage: {phase1_data["Coverage"].mean()*100:.2f}%\n')
        if 'Success_Rate' in phase1_data.columns:
            f.write(f'Average success rate: {phase1_data["Success_Rate"].mean()*100:.2f}%\n')
        f.write('\n')
        
        # Strategy comparison
        f.write('STRATEGY COMPARISON\n')
        f.write('------------------\n')
        for strategy in phase1_data['Strategy'].unique():
            strategy_data = phase1_data[phase1_data['Strategy'] == strategy]
            f.write(f'Strategy: {strategy}\n')
            f.write(f'  Completion Time: {strategy_data["Completion_Time"].mean():.2f} ± {strategy_data["Completion_Time"].std():.2f}\n')
            f.write(f'  Coverage: {strategy_data["Coverage"].mean()*100:.2f}% ± {strategy_data["Coverage"].std()*100:.2f}%\n')
            if 'Success_Rate' in phase1_data.columns:
                f.write(f'  Success Rate: {strategy_data["Success_Rate"].mean()*100:.2f}% ± {strategy_data["Success_Rate"].std()*100:.2f}%\n')
            f.write('\n')
        
        # Agent count comparison
        f.write('AGENT COUNT COMPARISON\n')
        f.write('---------------------\n')
        for count in sorted(phase1_data['Agent_Count'].unique()):
            count_data = phase1_data[phase1_data['Agent_Count'] == count]
            f.write(f'Agent Count: {count}\n')
            f.write(f'  Completion Time: {count_data["Completion_Time"].mean():.2f} ± {count_data["Completion_Time"].std():.2f}\n')
            f.write(f'  Coverage: {count_data["Coverage"].mean()*100:.2f}% ± {count_data["Coverage"].std()*100:.2f}%\n')
            if 'Success_Rate' in phase1_data.columns:
                f.write(f'  Success Rate: {count_data["Success_Rate"].mean()*100:.2f}% ± {count_data["Success_Rate"].std()*100:.2f}%\n')
            f.write('\n')
    
    print('Phase 1 analysis complete.')
except Exception as e:
    print(f'Error analyzing Phase 1: {e}')

# PART 2: ANALYZE PHASE 2 (ENVIRONMENTAL IMPACT)
print('Analyzing Phase 2: Environmental Impact...')
phase2_dir = os.path.join(output_dir, 'phase2')
ensure_dir(phase2_dir)

try:
    # Load Phase 2 data
    phase2_data = pd.read_csv(phase2_path)
    
    # Create boxplots for environmental impact
    create_boxplot(
        phase2_data, 'Obstacle_Density', 'Completion_Time', 'Strategy',
        'Completion Time by Obstacle Density and Strategy',
        os.path.join('phase2', 'completion_time_by_obstacle_density.png')
    )
    
    create_boxplot(
        phase2_data, 'Obstacle_Density', 'Coverage', 'Strategy',
        'Coverage by Obstacle Density and Strategy',
        os.path.join('phase2', 'coverage_by_obstacle_density.png')
    )
    
    if 'Success_Rate' in phase2_data.columns:
        create_boxplot(
            phase2_data, 'Obstacle_Density', 'Success_Rate', 'Strategy',
            'Success Rate by Obstacle Density and Strategy',
            os.path.join('phase2', 'success_rate_by_obstacle_density.png')
        )
    
    # Create line plots for strategy performance across obstacle densities
    create_lineplot(
        phase2_data, 'Obstacle_Density', 'Completion_Time', 'Strategy',
        'Strategy Performance Across Environment Complexity',
        os.path.join('phase2', 'strategy_performance_across_complexity.png')
    )
    
    create_lineplot(
        phase2_data, 'Strategy', 'Completion_Time', 'Obstacle_Density',
        'Environmental Impact by Strategy',
        os.path.join('phase2', 'environmental_impact_by_strategy.png')
    )
    
    # Generate summary statistics
    with open(os.path.join(phase2_dir, 'summary_statistics.txt'), 'w') as f:
        f.write('PHASE 2: ENVIRONMENTAL IMPACT SUMMARY\n')
        f.write('===================================\n\n')
        
        # Overall statistics
        f.write('OVERALL STATISTICS\n')
        f.write('-----------------\n')
        f.write(f'Total trials: {len(phase2_data)}\n')
        f.write(f'Average completion time: {phase2_data["Completion_Time"].mean():.2f} steps\n')
        f.write(f'Average coverage: {phase2_data["Coverage"].mean()*100:.2f}%\n')
        if 'Success_Rate' in phase2_data.columns:
            f.write(f'Average success rate: {phase2_data["Success_Rate"].mean()*100:.2f}%\n')
        f.write('\n')
        
        # Strategy comparison
        f.write('STRATEGY COMPARISON\n')
        f.write('------------------\n')
        for strategy in phase2_data['Strategy'].unique():
            strategy_data = phase2_data[phase2_data['Strategy'] == strategy]
            f.write(f'Strategy: {strategy}\n')
            f.write(f'  Completion Time: {strategy_data["Completion_Time"].mean():.2f} ± {strategy_data["Completion_Time"].std():.2f}\n')
            f.write(f'  Coverage: {strategy_data["Coverage"].mean()*100:.2f}% ± {strategy_data["Coverage"].std()*100:.2f}%\n')
            if 'Success_Rate' in phase2_data.columns:
                f.write(f'  Success Rate: {strategy_data["Success_Rate"].mean()*100:.2f}% ± {strategy_data["Success_Rate"].std()*100:.2f}%\n')
            f.write('\n')
        
        # Obstacle density comparison
        f.write('OBSTACLE DENSITY COMPARISON\n')
        f.write('-------------------------\n')
        for density in sorted(phase2_data['Obstacle_Density'].unique()):
            density_data = phase2_data[phase2_data['Obstacle_Density'] == density]
            f.write(f'Obstacle Density: {density}\n')
            f.write(f'  Completion Time: {density_data["Completion_Time"].mean():.2f} ± {density_data["Completion_Time"].std():.2f}\n')
            f.write(f'  Coverage: {density_data["Coverage"].mean()*100:.2f}% ± {density_data["Coverage"].std()*100:.2f}%\n')
            if 'Success_Rate' in phase2_data.columns:
                f.write(f'  Success Rate: {density_data["Success_Rate"].mean()*100:.2f}% ± {density_data["Success_Rate"].std()*100:.2f}%\n')
            f.write('\n')
        
        # Strategy performance at different obstacle densities
        f.write('STRATEGY PERFORMANCE BY OBSTACLE DENSITY\n')
        f.write('--------------------------------------\n')
        pivot = phase2_data.pivot_table(index='Strategy', columns='Obstacle_Density', 
                                      values='Completion_Time', aggfunc='mean')
        f.write(f'{pivot}\n\n')
    
    print('Phase 2 analysis complete.')
except Exception as e:
    print(f'Error analyzing Phase 2: {e}')

# PART 3: ANALYZE PHASE 3 (COMMUNICATION CONSTRAINTS)
print('Analyzing Phase 3: Communication Constraints...')
phase3_dir = os.path.join(output_dir, 'phase3')
ensure_dir(phase3_dir)

try:
    # Load Phase 3 data
    phase3_data = pd.read_csv(phase3_path)
    
    # Convert 'inf' strings to actual infinity values if needed
    if 'Comm_Range' in phase3_data.columns:
        if isinstance(phase3_data['Comm_Range'].iloc[0], str):
            phase3_data['Comm_Range'] = phase3_data['Comm_Range'].replace('inf', float('inf'))
    
    # For plotting purposes, replace inf with a label
    plot_data = phase3_data.copy()
    if 'Comm_Range' in plot_data.columns:
        plot_data['Comm_Range_Label'] = plot_data['Comm_Range'].apply(
            lambda x: 'Unlimited' if x == float('inf') else str(x))
    
    # Create boxplots for communication range impact
    if 'Comm_Range_Label' in plot_data.columns:
        create_boxplot(
            plot_data, 'Comm_Range_Label', 'Completion_Time', 'Agent_Count',
            'Completion Time by Communication Range and Agent Count',
            os.path.join('phase3', 'completion_time_by_comm_range.png')
        )
        
        create_boxplot(
            plot_data, 'Comm_Range_Label', 'Coverage', 'Agent_Count',
            'Coverage by Communication Range and Agent Count',
            os.path.join('phase3', 'coverage_by_comm_range.png')
        )
        
        if 'Success_Rate' in plot_data.columns:
            create_boxplot(
                plot_data, 'Comm_Range_Label', 'Success_Rate', 'Agent_Count',
                'Success Rate by Communication Range and Agent Count',
                os.path.join('phase3', 'success_rate_by_comm_range.png')
            )
    
        # Create line plots for comm range effects
        def comm_range_sort(x):
            mapping = {'Unlimited': float('inf'), '5.0': 5.0, '2.0': 2.0}
            return x.map(lambda v: mapping.get(v, v))
        
        create_lineplot(
            plot_data, 'Comm_Range_Label', 'Completion_Time', 'Agent_Count',
            'Effect of Communication Range on Completion Time',
            os.path.join('phase3', 'comm_range_effect_on_completion.png'),
            sort_key=comm_range_sort
        )
        
        create_lineplot(
            plot_data, 'Agent_Count', 'Completion_Time', 'Comm_Range_Label',
            'Effect of Agent Count on Completion Time by Communication Range',
            os.path.join('phase3', 'agent_count_effect_by_comm_range.png')
        )
    
    # Generate summary statistics
    with open(os.path.join(phase3_dir, 'summary_statistics.txt'), 'w') as f:
        f.write('PHASE 3: COMMUNICATION CONSTRAINTS SUMMARY\n')
        f.write('=========================================\n\n')
        
        # Overall statistics
        f.write('OVERALL STATISTICS\n')
        f.write('-----------------\n')
        f.write(f'Total trials: {len(phase3_data)}\n')
        f.write(f'Average completion time: {phase3_data["Completion_Time"].mean():.2f} steps\n')
        f.write(f'Average coverage: {phase3_data["Coverage"].mean()*100:.2f}%\n')
        if 'Success_Rate' in phase3_data.columns:
            f.write(f'Average success rate: {phase3_data["Success_Rate"].mean()*100:.2f}%\n')
        f.write('\n')
        
        # By communication range
        if 'Comm_Range' in phase3_data.columns:
            f.write('COMMUNICATION RANGE COMPARISON\n')
            f.write('-----------------------------\n')
            
            for comm_range in sorted(phase3_data['Comm_Range'].unique(), key=lambda x: float('-inf') if x == float('inf') else float(x)):
                range_data = phase3_data[phase3_data['Comm_Range'] == comm_range]
                
                if comm_range == float('inf'):
                    range_label = 'Unlimited'
                else:
                    range_label = str(comm_range)
                
                f.write(f'Communication Range: {range_label}\n')
                f.write(f'  Completion Time: {range_data["Completion_Time"].mean():.2f} ± {range_data["Completion_Time"].std():.2f}\n')
                f.write(f'  Coverage: {range_data["Coverage"].mean()*100:.2f}% ± {range_data["Coverage"].std()*100:.2f}%\n')
                if 'Success_Rate' in phase3_data.columns:
                    f.write(f'  Success Rate: {range_data["Success_Rate"].mean()*100:.2f}% ± {range_data["Success_Rate"].std()*100:.2f}%\n')
                f.write('\n')
        
        # By agent count
        f.write('AGENT COUNT COMPARISON\n')
        f.write('---------------------\n')
        for count in sorted(phase3_data['Agent_Count'].unique()):
            count_data = phase3_data[phase3_data['Agent_Count'] == count]
            f.write(f'Agent Count: {count}\n')
            f.write(f'  Completion Time: {count_data["Completion_Time"].mean():.2f} ± {count_data["Completion_Time"].std():.2f}\n')
            f.write(f'  Coverage: {count_data["Coverage"].mean()*100:.2f}% ± {count_data["Coverage"].std()*100:.2f}%\n')
            if 'Success_Rate' in phase3_data.columns:
                f.write(f'  Success Rate: {count_data["Success_Rate"].mean()*100:.2f}% ± {count_data["Success_Rate"].std()*100:.2f}%\n')
            f.write('\n')
        
        # Interaction effects
        if 'Comm_Range' in phase3_data.columns:
            f.write('INTERACTION EFFECTS\n')
            f.write('------------------\n')
            f.write('Agent Count x Communication Range on Completion Time:\n')
            
            # Create a pivot table for the interaction
            means_table = phase3_data.pivot_table(
                index='Agent_Count', 
                columns='Comm_Range', 
                values='Completion_Time', 
                aggfunc='mean'
            )
            f.write(f'{means_table}\n\n')
    
    print('Phase 3 analysis complete.')
except Exception as e:
    print(f'Error analyzing Phase 3: {e}')

# PART 4: COMBINED ANALYSIS ACROSS PHASES
print('Creating combined analysis...')
combined_dir = os.path.join(output_dir, 'combined')
ensure_dir(combined_dir)

try:
    # Load data from all phases
    all_data = []
    try:
        phase1_data = pd.read_csv(phase1_path)
        phase1_data['Phase'] = 1
        all_data.append(phase1_data)
    except:
        print('Warning: Could not load Phase 1 data')
    
    try:
        phase2_data = pd.read_csv(phase2_path)
        phase2_data['Phase'] = 2
        all_data.append(phase2_data)
    except:
        print('Warning: Could not load Phase 2 data')
    
    try:
        phase3_data = pd.read_csv(phase3_path)
        phase3_data['Phase'] = 3
        all_data.append(phase3_data)
    except:
        print('Warning: Could not load Phase 3 data')
    
    if all_data:
        # Combine data from all phases
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Create strategy comparison across all phases
        plt.figure(figsize=(12, 8))
        strategy_data = combined_data.groupby(['Phase', 'Strategy'])['Completion_Time'].mean().reset_index()
        
        # Only proceed if we have multiple strategies
        if len(strategy_data['Strategy'].unique()) > 1:
            for strategy in strategy_data['Strategy'].unique():
                strat_data = strategy_data[strategy_data['Strategy'] == strategy]
                plt.plot(strat_data['Phase'], strat_data['Completion_Time'], marker='o', linewidth=2, label=strategy)
            
            plt.title('Strategy Performance Across Experimental Phases')
            plt.xlabel('Experimental Phase')
            plt.ylabel('Average Completion Time')
            plt.xticks([1, 2, 3], ['Phase 1\nAgent Count', 'Phase 2\nEnvironment', 'Phase 3\nCommunication'])
            plt.legend(title='Strategy')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(combined_dir, 'strategy_across_phases.png'), dpi=300)
            plt.close()
        
        # Create agent count comparison across phases
        plt.figure(figsize=(12, 8))
        agent_data = combined_data.groupby(['Phase', 'Agent_Count'])['Completion_Time'].mean().reset_index()
        
        for count in sorted(agent_data['Agent_Count'].unique()):
            count_data = agent_data[agent_data['Agent_Count'] == count]
            plt.plot(count_data['Phase'], count_data['Completion_Time'], marker='o', linewidth=2, label=str(count))
        
        plt.title('Agent Count Performance Across Experimental Phases')
        plt.xlabel('Experimental Phase')
        plt.ylabel('Average Completion Time')
        plt.xticks([1, 2, 3], ['Phase 1\nAgent Count', 'Phase 2\nEnvironment', 'Phase 3\nCommunication'])
        plt.legend(title='Agent Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, 'agent_count_across_phases.png'), dpi=300)
        plt.close()
        
        # Generate overall summary statistics
        with open(os.path.join(combined_dir, 'overall_summary.txt'), 'w') as f:
            f.write('OVERALL EXPERIMENT SUMMARY\n')
            f.write('=========================\n\n')
            
            f.write('EXPERIMENTAL OVERVIEW\n')
            f.write('-------------------\n')
            f.write('Phase 1: Agent Count and Strategy Comparison\n')
            f.write('Phase 2: Environmental Impact\n')
            f.write('Phase 3: Communication Constraints\n\n')
            
            f.write(f'Total trials across all phases: {len(combined_data)}\n\n')
            
            f.write('STRATEGY COMPARISON SUMMARY\n')
            f.write('-------------------------\n')
            for strategy in combined_data['Strategy'].unique():
                strategy_data = combined_data[combined_data['Strategy'] == strategy]
                f.write(f'Strategy: {strategy}\n')
                f.write(f'  Overall Performance:\n')
                f.write(f'    Completion Time: {strategy_data["Completion_Time"].mean():.2f} ± {strategy_data["Completion_Time"].std():.2f}\n')
                f.write(f'    Coverage: {strategy_data["Coverage"].mean()*100:.2f}% ± {strategy_data["Coverage"].std()*100:.2f}%\n')
                if 'Success_Rate' in combined_data.columns:
                    f.write(f'    Success Rate: {strategy_data["Success_Rate"].mean()*100:.2f}% ± {strategy_data["Success_Rate"].std()*100:.2f}%\n')
                f.write('\n')
                
                f.write('  Performance by Phase:\n')
                for phase in sorted(strategy_data['Phase'].unique()):
                    phase_strat_data = strategy_data[strategy_data['Phase'] == phase]
                    f.write(f'    Phase {phase}: Completion Time = {phase_strat_data["Completion_Time"].mean():.2f}, Coverage = {phase_strat_data["Coverage"].mean()*100:.2f}%\n')
                f.write('\n')
            
            f.write('KEY FINDINGS\n')
            f.write('-----------\n')
            f.write('1. Agent Count Effects:\n')
            agent_effects = combined_data.groupby('Agent_Count')['Completion_Time'].mean().sort_values()
            best_agent_count = agent_effects.index[0]
            f.write(f'   - Optimal agent count across all experiments: {best_agent_count}\n')
            f.write(f'   - Performance ranking by agent count (best to worst): {", ".join(map(str, agent_effects.index))}\n\n')
            
            f.write('2. Strategy Effects:\n')
            strategy_effects = combined_data.groupby('Strategy')['Completion_Time'].mean().sort_values()
            best_strategy = strategy_effects.index[0]
            f.write(f'   - Best performing strategy across all experiments: {best_strategy}\n')
            f.write(f'   - Performance ranking by strategy (best to worst): {", ".join(map(str, strategy_effects.index))}\n\n')
            
            if 'Obstacle_Density' in combined_data.columns:
                f.write('3. Environmental Effects:\n')
                env_effects = combined_data.groupby('Obstacle_Density')['Completion_Time'].mean().sort_values()
                f.write(f'   - Performance ranking by obstacle density (best to worst): {", ".join(map(str, env_effects.index))}\n\n')
            
            if 'Comm_Range' in combined_data.columns:
                f.write('4. Communication Effects:\n')
                if isinstance(combined_data['Comm_Range'].iloc[0], str):
                    combined_data['Comm_Range_Numeric'] = combined_data['Comm_Range'].replace('inf', float('inf'))
                else:
                    combined_data['Comm_Range_Numeric'] = combined_data['Comm_Range']
                
                comm_effects = combined_data.groupby('Comm_Range_Numeric')['Completion_Time'].mean()
                # Sort by comm range, handling inf specially
                comm_effects = comm_effects.reset_index().sort_values('Comm_Range_Numeric', 
                                                             key=lambda x: x.map(lambda v: float('-inf') if v == float('inf') else float(v)))
                
                # Format for output
                comm_ranges_formatted = []
                for cr in comm_effects['Comm_Range_Numeric']:
                    if cr == float('inf'):
                        comm_ranges_formatted.append('Unlimited')
                    else:
                        comm_ranges_formatted.append(str(cr))
                
                f.write(f'   - Performance ranking by communication range (best to worst): {", ".join(comm_ranges_formatted)}\n\n')
            
            f.write('CONCLUSIONS AND RECOMMENDATIONS\n')
            f.write('-------------------------------\n')
            f.write(f'Based on the experimental results, the {best_strategy} strategy with {best_agent_count} agents appears to offer the best balance of efficiency and coverage.\n\n')
            
            # Try to identify if there are special cases
            if 'Obstacle_Density' in combined_data.columns and len(combined_data['Strategy'].unique()) > 1:
                # Check if different strategies perform better in different environments
                best_by_env = combined_data.groupby(['Obstacle_Density', 'Strategy'])['Completion_Time'].mean().reset_index()
                best_by_env = best_by_env.sort_values(['Obstacle_Density', 'Completion_Time'])
                best_strategies_by_env = best_by_env.groupby('Obstacle_Density').first().reset_index()
                
                if len(best_strategies_by_env['Strategy'].unique()) > 1:
                    f.write('Special consideration: Different strategies perform better in different environments:\n')
                    for _, row in best_strategies_by_env.iterrows():
                        f.write(f'  - {row["Strategy"]} is best for obstacle density {row["Obstacle_Density"]}\n')
            
            f.write('\nFurther research could explore adaptive strategies that select the appropriate coordination method based on environmental conditions and available resources.\n')
    
    print('Combined analysis complete.')
except Exception as e:
    print(f'Error creating combined analysis: {e}')

print(f'All analyses completed. Results saved to {output_dir}')