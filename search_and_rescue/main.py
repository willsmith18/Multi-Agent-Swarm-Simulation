"""
Main module for the Search and Rescue simulation.
This is the entry point for running the simulations.
"""

import tkinter as tk
import argparse
import sys
import os
import time
import numpy as np
import csv
from datetime import datetime

from environment import DisasterEnvironment
from agents import Agent, StigmergyAgent, CommunicatingAgent
from controller import ExperimentController

def parse_arguments():
    """Parse command line arguments for automated experiments"""
    parser = argparse.ArgumentParser(description='Search and Rescue Simulation')
    
    parser.add_argument('--batch', action='store_true', 
                        help='Run batch simulations without GUI')
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
    parser.add_argument('--trials', type=int, default=5,
                        help='Number of trials for batch mode (default: 5)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per trial (default: 1000)')
    parser.add_argument('--output', type=str, default='experiment_results',
                        help='Output directory for results (default: experiment_results)')
    
    return parser.parse_args()

def run_batch_experiment(args):
    """Run batch experiments without GUI"""
    print(f"Running batch experiment with {args.strategy} strategy and {args.agents} agents")
    print(f"Grid size: {args.grid_size}, Obstacle density: {args.obstacle_density}, Victims: {args.victims}")
    print(f"Trials: {args.trials}, Max steps: {args.max_steps}")
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Results storage
    results = {
        'completion_times': [],
        'coverage': [],
        'path_lengths': [],
        'idle_times': [],
        'victims_rescued': [],
        'victims_total': []
    }
    
    # Run trials
    for trial in range(args.trials):
        print(f"\nTrial {trial+1}/{args.trials}")
        
        # Create environment and agents
        env = DisasterEnvironment(args.grid_size)
        env.create_obstacles(args.obstacle_density)
        env.create_disaster_zone(args.victims)
        
        agents = []
        for i in range(args.agents):
            if args.strategy == 'basic':
                agent = Agent(f"Agent{i}", env)
            elif args.strategy == 'stigmergy':
                agent = StigmergyAgent(f"Agent{i}", env)
            elif args.strategy == 'communication':
                agent = CommunicatingAgent(f"Agent{i}", env)
            
            agents.append(agent)
        
        # Run simulation
        step_count = 0
        env.start_mission()
        
        while step_count < args.max_steps:
            # Update all agents
            for agent in agents:
                agent.update(agents)
            
            step_count += 1
            
            # Check if mission is complete
            if env.is_mission_complete():
                break
        
        # Record metrics
        results['completion_times'].append(step_count)
        results['victims_rescued'].append(env.victims_rescued)
        results['victims_total'].append(env.victims_total)
        
        # Calculate coverage
        all_visited = set()
        for agent in agents:
            all_visited.update(agent.visited_cells)
        
        total_cells = env.grid_size * env.grid_size
        obstacle_count = np.sum(env.grid == 1)
        navigable_cells = total_cells - obstacle_count
        coverage = len(all_visited) / max(1, navigable_cells)
        
        results['coverage'].append(coverage)
        
        # Record agent metrics
        trial_path_lengths = []
        trial_idle_times = []
        
        for agent in agents:
            trial_path_lengths.append(agent.path_length)
            trial_idle_times.append(agent.idle_time)
        
        results['path_lengths'].append(trial_path_lengths)
        results['idle_times'].append(trial_idle_times)
        
        print(f"Completed in {step_count} steps")
        print(f"Victims rescued: {env.victims_rescued}/{env.victims_total}")
        print(f"Coverage: {coverage*100:.1f}%")
        print(f"Average path length: {np.mean(trial_path_lengths):.1f}")
        print(f"Average idle time: {np.mean(trial_idle_times):.1f}")
    
    # Save results
    save_batch_results(args, results)
    
    # Print summary
    print("\n\nBATCH EXPERIMENT SUMMARY")
    print("========================")
    print(f"Strategy: {args.strategy}")
    print(f"Agent count: {args.agents}")
    print(f"Trials: {args.trials}")
    print(f"Average completion time: {np.mean(results['completion_times']):.1f} ± {np.std(results['completion_times']):.1f}")
    print(f"Average coverage: {np.mean(results['coverage'])*100:.1f}% ± {np.std(results['coverage'])*100:.1f}%")
    avg_path = np.mean([np.mean(paths) for paths in results['path_lengths']])
    print(f"Average path length: {avg_path:.1f}")
    avg_idle = np.mean([np.mean(idles) for idles in results['idle_times']])
    print(f"Average idle time: {avg_idle:.1f}")
    print(f"Average victims rescued: {np.mean(results['victims_rescued']):.1f}/{np.mean(results['victims_total']):.1f}")

def save_batch_results(args, results):
    """Save batch experiment results to CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.output}/batch_{args.strategy}_{args.agents}agents_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write experiment parameters
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['Strategy', args.strategy])
        writer.writerow(['Agent Count', args.agents])
        writer.writerow(['Grid Size', args.grid_size])
        writer.writerow(['Obstacle Density', args.obstacle_density])
        writer.writerow(['Victim Count Target', args.victims])
        writer.writerow(['Trials', args.trials])
        writer.writerow(['Max Steps', args.max_steps])
        writer.writerow([])
        
        # Write trial results
        writer.writerow(['Trial', 'Completion Time', 'Coverage', 'Avg Path Length', 
                       'Avg Idle Time', 'Victims Rescued', 'Victims Total'])
        
        for i in range(args.trials):
            writer.writerow([
                i+1,
                results['completion_times'][i],
                results['coverage'][i],
                np.mean(results['path_lengths'][i]),
                np.mean(results['idle_times'][i]),
                results['victims_rescued'][i],
                results['victims_total'][i]
            ])
    
    print(f"Results saved to {filename}")

def main():
    """Main function for the search and rescue simulation"""
    args = parse_arguments()
    
    if args.batch:
        # Run batch mode without GUI
        run_batch_experiment(args)
    else:
        # Run interactive mode with GUI
        window = tk.Tk()
        controller = ExperimentController(window)
        
        # Apply command line arguments if provided
        controller.strategy_var.set(args.strategy)
        controller.agent_count_var.set(args.agents)
        controller.obstacle_density_var.set(args.obstacle_density)
        controller.victim_count_var.set(args.victims)
        
        # Setup environment and agents
        controller.setup_environment()
        controller.create_agents(args.agents, args.strategy)
        
        # Draw initial state
        controller.draw()
        
        # Start the main loop
        window.mainloop()

if __name__ == "__main__":
    main()