"""
Main module for the Search and Rescue simulation.
"""

import tkinter as tk
import argparse
import sys
import os
import time
import numpy as np
import csv
import random
from datetime import datetime

from environment import DisasterEnvironment
from agents import Agent, StigmergyAgent, CommunicatingAgent
from controller import ExperimentController

def parse_arguments():
    """Parse command line arguments for simulation and experiments"""
    parser = argparse.ArgumentParser(description='Search and Rescue Simulation')
    
    # Main command mode
    parser.add_argument('--mode', type=str, choices=['gui', 'batch', 'phase1', 'phase2', 'phase3', 'analyze'],
                        default='gui', help='Operation mode (default: gui)')
    
    # GUI simulation options
    parser.add_argument('--strategy', type=str, choices=['basic', 'stigmergy', 'communication'],
                        default='basic', help='Coordination strategy (default: basic)')
    parser.add_argument('--agents', type=int, default=3,
                        help='Number of agents (default: 3)')
    parser.add_argument('--grid_size', type=int, default=20,
                        help='Grid size (default: 20)')
    parser.add_argument('--obstacle_density', type=float, default=0.15,
                        help='Obstacle density (default: 0.15)')
    parser.add_argument('--victims', type=int, default=10,
                        help='Number of victims (default: 10)')
    
    # Experiment options
    parser.add_argument('--repetitions', type=int, default=None,
                        help='Number of repetitions for experiments (default: Phase 1=5, Phase 2=3, Phase 3=1)')
    parser.add_argument('--comm_range', type=float, default=5.0,
                        help='Communication range for communication strategy (default: 5.0)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per trial (default: 1000)')
    parser.add_argument('--output', type=str, default='experiment_results',
                        help='Output directory for results (default: experiment_results)')
    
    # Analysis options
    parser.add_argument('--files', nargs='+',
                        help='CSV result files to analyze (required for analysis mode)')
    
    return parser.parse_args()

def run_gui_mode(args):
    """Run the simulation in GUI mode"""
    window = tk.Tk()
    controller = ExperimentController(window)
    
    # Apply command line arguments if provided
    controller.strategy_var.set(args.strategy)
    controller.agent_count_var.set(args.agents)
    controller.obstacle_density_var.set(args.obstacle_density)
    controller.victim_count_var.set(args.victims)
    
    # Setup environment and agents
    controller.setup_environment()
    controller.create_agents(args.agents, args.strategy, args.comm_range)
    
    # Draw initial state
    controller.draw()
    
    # Start the main loop
    window.mainloop()
    
def run_batch_mode(args):
    """Run batch experiments without GUI"""
    print(f"Running batch experiment with {args.strategy} strategy and {args.agents} agents")
    print(f"Grid size: {args.grid_size}, Obstacle density: {args.obstacle_density}, Victims: {args.victims}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create headless controller
    controller = ExperimentController(window=None)
    controller.results_directory = args.output
    controller.max_steps = args.max_steps
    
    # Set experiment parameters
    repetitions = args.repetitions if args.repetitions is not None else 5
    
    # Run simulation with specified parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.output}/batch_{args.strategy}_{args.agents}agents_{timestamp}.csv"
    
    # Create CSV file and write header
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Strategy', 'Agent_Count', 'Obstacle_Density', 'Victim_Count', 
            'Trial', 'Completion_Time', 'Coverage', 'Avg_Path_Length', 
            'Avg_Idle_Time', 'Victims_Rescued', 'Victims_Total', 'Success_Rate'
        ])
    
    # Run trials
    for rep in range(repetitions):
        print(f"\nTrial {rep+1}/{repetitions}")
        
        # Reset controller for each trial
        controller.reset_simulation()
        
        # Setup environment
        controller.environment = DisasterEnvironment(args.grid_size)
        controller.environment.create_obstacles(args.obstacle_density)
        controller.environment.create_disaster_zone(args.victims)
        
        # Create agents
        controller.create_agents(args.agents, args.strategy, args.comm_range)
        
        # Run simulation
        controller.running = True
        controller.step_count = 0
        controller.environment.start_mission()
        
        while controller.running and controller.step_count < controller.max_steps:
            # Record current victims rescued
            prev_rescued = controller.environment.victims_rescued
            
            # Update all agents
            for agent in controller.agents:
                agent.update(controller.agents)
            
            # Check if new victims were rescued
            if controller.environment.victims_rescued > prev_rescued:
                controller.metrics['rescue_times'].append(controller.step_count)
            
            # Record rescue rate
            controller.metrics['rescue_rate'].append(
                controller.environment.victims_rescued / max(1, controller.environment.victims_total)
            )
            
            # Increment step count
            controller.step_count += 1
            
            # Check if mission is complete
            if controller.environment.is_mission_complete() or controller.step_count >= controller.max_steps:
                controller.running = False
        
        # Record metrics
        controller.record_final_metrics()
        
        # Calculate success rate
        success_rate = controller.environment.victims_rescued / max(1, controller.environment.victims_total)
        
        # Write results to CSV
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                args.strategy,
                args.agents,
                args.obstacle_density,
                args.victims,
                rep + 1,
                controller.metrics['completion_time'],
                controller.metrics['coverage'],
                np.mean(controller.metrics['path_lengths']),
                np.mean(controller.metrics['idle_times']),
                controller.environment.victims_rescued,
                controller.environment.victims_total,
                success_rate
            ])
    
    print(f"\nBatch experiments completed. Results saved to {filename}")
    return filename

def run_phase_experiment(args, phase):
    """Run a specific experimental phase"""
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create headless controller 
    controller = ExperimentController(window=None)
    controller.results_directory = args.output
    controller.max_steps = args.max_steps
    
    # Set default repetitions if not specified
    if args.repetitions is None:
        if phase == 1:
            repetitions = 5
        elif phase == 2:
            repetitions = 3
        else:  # phase == 3
            repetitions = 1
    else:
        repetitions = args.repetitions
    
    # Run the specified phase
    if phase == 1:
        filename = controller.run_phase1(repetitions)
    elif phase == 2:
        filename = controller.run_phase2(repetitions)
    elif phase == 3:
        filename = controller.run_phase3(repetitions)
    else:
        print(f"Unknown phase: {phase}")
        return None
    
    return filename

def analyze_results(args):
    """Analyze experiment results"""
    if not args.files:
        print("ERROR: No files to analyze. Please specify files with --files option.")
        return
    
    # Check that all files exist
    for file in args.files:
        if not os.path.exists(file):
            print(f"ERROR: File not found: {file}")
            return
    
    # Create visualization
    try:
        # Try to import visualization utilities
        from visualization_all import create_comprehensive_report
        output_dir = create_comprehensive_report(args.files, args.output)
        print(f"Analysis complete. Results saved to: {output_dir}")
    except ImportError:
        print("ERROR: Visualization utilities not found. Please ensure visualization_utils.py is in your path.")
        # Create basic analysis
        analyze_results_basic(args.files, args.output)

def analyze_results_basic(files, output_dir):
    """Create a basic analysis of results without visualization_utils"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Combine data
    data = pd.concat(dfs, ignore_index=True)
    
    # Create basic summary statistics
    with open(f"{output_dir}/summary_statistics.txt", 'w') as f:
        f.write("SEARCH AND RESCUE EXPERIMENT SUMMARY\n")
        f.write("===================================\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"Total trials: {len(data)}\n")
        if 'Completion_Time' in data.columns:
            f.write(f"Average completion time: {data['Completion_Time'].mean():.2f} steps\n")
        if 'Coverage' in data.columns:
            f.write(f"Average coverage: {data['Coverage'].mean()*100:.2f}%\n")
        if 'Success_Rate' in data.columns:
            f.write(f"Average success rate: {data['Success_Rate'].mean()*100:.2f}%\n\n")
        
        # Strategy comparison if applicable
        if 'Strategy' in data.columns:
            f.write("STRATEGY COMPARISON\n")
            f.write("------------------\n")
            for strategy in data['Strategy'].unique():
                strategy_data = data[data['Strategy'] == strategy]
                f.write(f"Strategy: {strategy}\n")
                if 'Completion_Time' in data.columns:
                    f.write(f"  Completion Time: {strategy_data['Completion_Time'].mean():.2f} ± {strategy_data['Completion_Time'].std():.2f}\n")
                if 'Coverage' in data.columns:
                    f.write(f"  Coverage: {strategy_data['Coverage'].mean()*100:.2f}% ± {strategy_data['Coverage'].std()*100:.2f}%\n")
                if 'Success_Rate' in data.columns:
                    f.write(f"  Success Rate: {strategy_data['Success_Rate'].mean()*100:.2f}% ± {strategy_data['Success_Rate'].std()*100:.2f}%\n")
                f.write("\n")
    
    # Create basic plots
    try:
        # Plot completion time by strategy and agent count
        if 'Strategy' in data.columns and 'Agent_Count' in data.columns and 'Completion_Time' in data.columns:
            plt.figure(figsize=(10, 6))
            strategies = data['Strategy'].unique()
            
            for strategy in strategies:
                strategy_data = data[data['Strategy'] == strategy]
                agent_counts = sorted(strategy_data['Agent_Count'].unique())
                
                completion_times = []
                for count in agent_counts:
                    times = strategy_data[strategy_data['Agent_Count'] == count]['Completion_Time']
                    completion_times.append(times.mean())
                
                plt.plot(agent_counts, completion_times, marker='o', label=strategy)
            
            plt.title('Completion Time by Strategy and Agent Count')
            plt.xlabel('Number of Agents')
            plt.ylabel('Average Completion Time (steps)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/completion_time_by_agents.png")
            
        # Plot coverage by strategy
        if 'Strategy' in data.columns and 'Coverage' in data.columns:
            plt.figure(figsize=(8, 6))
            strategy_coverage = data.groupby('Strategy')['Coverage'].mean() * 100
            strategy_coverage.plot(kind='bar')
            plt.title('Average Coverage by Strategy')
            plt.xlabel('Strategy')
            plt.ylabel('Coverage (%)')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/coverage_by_strategy.png")
    except Exception as e:
        print(f"Warning: Could not create plots. Error: {e}")
    
    print(f"Basic analysis complete. Results saved to: {output_dir}")

def main():
    """Main function for the search and rescue simulation"""
    args = parse_arguments()
    
    # Run in the specified mode
    if args.mode == 'gui':
        run_gui_mode(args)
    elif args.mode == 'batch':
        run_batch_mode(args)
    elif args.mode == 'phase1':
        run_phase_experiment(args, 1)
    elif args.mode == 'phase2':
        run_phase_experiment(args, 2)
    elif args.mode == 'phase3':
        run_phase_experiment(args, 3)
    elif args.mode == 'analyze':
        analyze_results(args)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()