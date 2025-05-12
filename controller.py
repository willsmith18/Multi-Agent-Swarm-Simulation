"""
Controller module for Search and Rescue simulation.
Manages experiment setup, execution, and metrics collection.
"""

import tkinter as tk
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import os
import random
from datetime import datetime
import argparse
from tqdm import tqdm

from environment import DisasterEnvironment
from agents import Agent, StigmergyAgent, CommunicatingAgent

class ExperimentController:
    """
    Manages experiment setup, execution, and metrics for the search and rescue simulation.
    Supports different coordination strategies and varying numbers of agents.
    """
    
    def __init__(self, window=None, grid_size=20, cell_size=30):
        self.window = window
        
        # Create environment
        self.environment = DisasterEnvironment(grid_size, cell_size)
        
        # GUI components
        if window:
            self.window.title("Search and Rescue Simulation")
            
            # Canvas setup
            canvas_width = self.environment.width
            canvas_height = self.environment.height + 60  # Extra space for metrics
            self.canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
            self.canvas.pack()
            
            # Create UI controls
            self.create_controls()
            
            # Results visualization
            self.fig = None
            self.create_results_window()
        else:
            self.canvas = None
        
        # Agents
        self.agents = []
        
        # Experiment parameters
        self.running = False
        self.update_interval = 100  # milliseconds
        self.step_count = 0
        self.max_steps = 1000 
        self.current_strategy = "basic"
        
        # Performance metrics
        self.metrics = {
            'rescue_times': [],       # Time steps when victims were rescued
            'completion_time': None,  # Total time steps to rescue all victims
            'path_lengths': [],       # Total path length for each agent
            'idle_times': [],         # Idle time for each agent
            'coverage': 0,            # Percentage of environment explored
            'rescue_rate': []         # Victims rescued over time
        }
        
        # Results storage for batch experiments
        self.results_directory = "experiment_results"
        os.makedirs(self.results_directory, exist_ok=True)
    
    def create_controls(self):
        """Create UI controls for the simulation"""
        control_frame = tk.Frame(self.window)
        control_frame.pack(fill=tk.X)
        
        # Start button
        self.start_button = tk.Button(control_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Reset button
        self.reset_button = tk.Button(control_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Coordination strategy selection
        self.strategy_var = tk.StringVar(value="basic")
        
        strategy_frame = tk.LabelFrame(control_frame, text="Coordination Strategy")
        strategy_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        basic_radio = tk.Radiobutton(strategy_frame, text="Basic", variable=self.strategy_var, 
                                    value="basic", command=self.update_strategy)
        basic_radio.pack(anchor=tk.W)
        
        stigmergy_radio = tk.Radiobutton(strategy_frame, text="Stigmergy", variable=self.strategy_var, 
                                        value="stigmergy", command=self.update_strategy)
        stigmergy_radio.pack(anchor=tk.W)
        
        communication_radio = tk.Radiobutton(strategy_frame, text="Communication", 
                                           variable=self.strategy_var, 
                                           value="communication", command=self.update_strategy)
        communication_radio.pack(anchor=tk.W)
        
        # Agent count
        agent_frame = tk.LabelFrame(control_frame, text="Simulation Parameters")
        agent_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Label(agent_frame, text="Agents:").grid(row=0, column=0)
        self.agent_count_var = tk.IntVar(value=3)
        agent_count = tk.Spinbox(agent_frame, from_=1, to=10, width=2, textvariable=self.agent_count_var)
        agent_count.grid(row=0, column=1)
        
        # Obstacle density
        tk.Label(agent_frame, text="Obstacle Density:").grid(row=1, column=0)
        self.obstacle_density_var = tk.DoubleVar(value=0.15)
        obstacle_density = tk.Spinbox(agent_frame, from_=0.0, to=0.3, increment=0.05, width=3, 
                                      textvariable=self.obstacle_density_var)
        obstacle_density.grid(row=1, column=1)
        
        # Victim count
        tk.Label(agent_frame, text="Victims:").grid(row=2, column=0)
        self.victim_count_var = tk.IntVar(value=10)
        victim_count = tk.Spinbox(agent_frame, from_=5, to=20, width=2, 
                                 textvariable=self.victim_count_var)
        victim_count.grid(row=2, column=1)
        
        # Batch experiment frame
        batch_frame = tk.LabelFrame(control_frame, text="Batch Experiments")
        batch_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Run multiple trials button
        self.run_trials_button = tk.Button(batch_frame, text="Run Trials", 
                                          command=self.run_batch_trials)
        self.run_trials_button.pack(pady=2)
        
        # Compare strategies button
        self.compare_strategies_button = tk.Button(batch_frame, text="Compare Strategies", 
                                                 command=self.compare_strategies)
        self.compare_strategies_button.pack(pady=2)
        
        # Export results button
        self.export_results_button = tk.Button(batch_frame, text="Export Results", 
                                             command=self.export_results)
        self.export_results_button.pack(pady=2)
        
        # New automated experiment buttons
        auto_frame = tk.LabelFrame(control_frame, text="Automated Experiments")
        auto_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Run automated phases
        self.run_phase1_button = tk.Button(auto_frame, text="Run Phase 1", 
                                          command=lambda: self.run_automated_phase(1))
        self.run_phase1_button.pack(pady=2)
        
        self.run_phase2_button = tk.Button(auto_frame, text="Run Phase 2", 
                                          command=lambda: self.run_automated_phase(2))
        self.run_phase2_button.pack(pady=2)
        
        self.run_phase3_button = tk.Button(auto_frame, text="Run Phase 3", 
                                          command=lambda: self.run_automated_phase(3))
        self.run_phase3_button.pack(pady=2)
    
    def create_results_window(self):
        """Create a window for displaying experiment results"""
        self.results_window = tk.Toplevel(self.window)
        self.results_window.title("Experiment Results")
        self.results_window.geometry("800x600")
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(2, 2, figsize=(8, 6))
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.results_window)
        self.canvas_widget = self.canvas_fig.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Hide window initially
        self.results_window.withdraw()
    
    def update_strategy(self):
        """Update coordination strategy for the simulation"""
        self.current_strategy = self.strategy_var.get()
    
    def setup_environment(self):
        """Create environment with obstacles and victims, ensuring all victims are accessible"""
        obstacle_density = self.obstacle_density_var.get() if hasattr(self, 'obstacle_density_var') else 0.15
        victim_count = self.victim_count_var.get() if hasattr(self, 'victim_count_var') else 10
        
        # Create obstacles
        self.environment.create_obstacles(obstacle_density=obstacle_density)
        
        # Create disaster zone with victims
        self.environment.create_disaster_zone(victim_count=victim_count)
        
        # Verify all victims are accessible
        all_accessible, inaccessible = self.environment.verify_victims_accessibility()
        
        if not all_accessible:
            print(f"Warning: Found {len(inaccessible)} inaccessible victims!")
            
            # Remove inaccessible victims
            for pos in inaccessible:
                self.environment.grid[pos[0]][pos[1]] = 0
                self.environment.victims_total -= 1
            
            # Try to place additional victims to reach the target count
            additional_needed = victim_count - self.environment.victims_total
            if additional_needed > 0:
                print(f"Attempting to place {additional_needed} additional victims...")
                attempts = 0
                max_attempts = 200
                placed = 0
                
                while placed < additional_needed and attempts < max_attempts:
                    x = random.randint(0, self.environment.grid_size - 1)
                    y = random.randint(0, self.environment.grid_size - 1)
                    
                    # Check if position is empty
                    if self.environment.grid[x][y] == 0:
                        # Check for adjacent empty cells
                        has_adjacent_empty = False
                        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.environment.grid_size and 
                                0 <= ny < self.environment.grid_size and 
                                self.environment.grid[nx][ny] == 0):
                                has_adjacent_empty = True
                                break
                        
                        if has_adjacent_empty:
                            self.environment.grid[x][y] = 2 
                            if self.environment._is_accessible((x, y)):
                                placed += 1
                                self.environment.victims_total += 1
                            else:
                                self.environment.grid[x][y] = 0 
                    
                    attempts += 1
                
                print(f"Successfully placed {placed} additional victims.")
            
            print(f"Final victim count: {self.environment.victims_total}")
    
    def create_agents(self, count=3, strategy="basic", comm_range=5):
        """Create and place agents in the environment based on strategy"""
        self.agents = []
        
        for i in range(count):
            if strategy == "basic":
                agent = Agent(f"Agent{i}", self.environment)
            elif strategy == "stigmergy":
                agent = StigmergyAgent(f"Agent{i}", self.environment)
            elif strategy == "communication":
                agent = CommunicatingAgent(f"Agent{i}", self.environment)
                # Set communication range if provided
                if hasattr(agent, 'comm_range'):
                    agent.comm_range = comm_range
            else:
                agent = Agent(f"Agent{i}", self.environment)
                
            self.agents.append(agent)
    
    def start_simulation(self):
        """Start or resume the simulation"""
        if not self.running:
            self.running = True
            # Only reset step count if starting a new simulation, not when resuming
            if self.step_count == 0:
                self.environment.start_mission()
                self.metrics['rescue_rate'] = []
            self.update_simulation()
            if hasattr(self, 'start_button'):
                self.start_button.config(text="Pause", command=self.pause_simulation)
    
    def pause_simulation(self):
        """Pause the simulation"""
        if self.running:
            self.running = False
            if hasattr(self, 'start_button'):
                self.start_button.config(text="Resume", command=self.start_simulation)
    
    def reset_simulation(self):
        """Reset the entire simulation"""
        self.running = False
        if hasattr(self, 'start_button'):
            self.start_button.config(text="Start", command=self.start_simulation, state=tk.NORMAL)
        self.step_count = 0
        
        # Reset metrics
        self.metrics = {
            'rescue_times': [],
            'completion_time': None,
            'path_lengths': [],
            'idle_times': [],
            'coverage': 0,
            'rescue_rate': []
        }
        
        # Reset environment
        self.environment = DisasterEnvironment(
            self.environment.grid_size, 
            self.environment.cell_size if hasattr(self.environment, 'cell_size') else 30
        )
        self.setup_environment()
        
        # Create new agents
        if hasattr(self, 'strategy_var') and hasattr(self, 'agent_count_var'):
            strategy = self.strategy_var.get()
            count = self.agent_count_var.get()
        else:
            strategy = self.current_strategy
            count = len(self.agents) if self.agents else 3
            
        self.create_agents(count, strategy)
        
        # Redraw if in GUI mode
        if self.canvas:
            self.draw()
    
    def update_simulation(self):
        """Update the simulation state"""
        if self.running:
            # Record current victims rescued
            prev_rescued = self.environment.victims_rescued
            
            # Update all agents
            for agent in self.agents:
                agent.update(self.agents)
            
            # Check if new victims were rescued
            if self.environment.victims_rescued > prev_rescued:
                self.metrics['rescue_times'].append(self.step_count)
            
            # Record rescue rate
            self.metrics['rescue_rate'].append(
                self.environment.victims_rescued / max(1, self.environment.victims_total)
            )
            
            # Increment step count
            self.step_count += 1
            
            # Redraw everything if in GUI mode
            if self.canvas:
                self.draw()
            
            # Check if mission is complete or max steps reached
            if self.environment.is_mission_complete() or self.step_count >= self.max_steps:
                self.running = False
                if hasattr(self, 'start_button'):
                    self.start_button.config(text="Completed", state=tk.DISABLED)
                
                # Record metrics
                self.record_final_metrics()
                
                # Display results if in GUI mode
                if self.canvas:
                    self.display_results()
                
                return
            
            # Schedule next update if in GUI mode
            if self.canvas and self.window:
                self.window.after(self.update_interval, self.update_simulation)
    
    def record_final_metrics(self):
        """Record the final metrics for the simulation"""
        # Record completion time
        self.metrics['completion_time'] = self.step_count
        
        # Record path lengths and idle times
        for agent in self.agents:
            self.metrics['path_lengths'].append(agent.path_length)
            self.metrics['idle_times'].append(agent.idle_time)
        
        # Calculate coverage (unique cells visited by any agent)
        all_visited = set()
        for agent in self.agents:
            all_visited.update(agent.visited_cells)
        
        total_cells = self.environment.grid_size * self.environment.grid_size
        obstacle_count = np.sum(self.environment.grid == 1)
        navigable_cells = total_cells - obstacle_count
        
        self.metrics['coverage'] = len(all_visited) / max(1, navigable_cells)
        
        # Print metrics to console
        print(f"Simulation completed in {self.step_count} steps")
        print(f"Strategy: {self.current_strategy}")
        print(f"Agents: {len(self.agents)}")
        print(f"Victims rescued: {self.environment.victims_rescued}/{self.environment.victims_total}")
        print(f"Coverage: {self.metrics['coverage']*100:.1f}%")
        print(f"Average path length: {np.mean(self.metrics['path_lengths']):.1f}")
        print(f"Average idle time: {np.mean(self.metrics['idle_times']):.1f}")
    
    def display_results(self):
        """Display the results visually"""
        # Show results window
        if hasattr(self, 'results_window'):
            self.results_window.deiconify()
            
            # Clear previous plots
            for ax_row in self.ax:
                for ax in ax_row:
                    ax.clear()
            
            # Plot rescue rate over time
            self.ax[0, 0].plot(self.metrics['rescue_rate'])
            self.ax[0, 0].set_title('Victim Rescue Rate')
            self.ax[0, 0].set_xlabel('Time Steps')
            self.ax[0, 0].set_ylabel('Fraction Rescued')
            self.ax[0, 0].grid(True)
            
            # Plot agent path lengths
            agent_ids = [f"Agent {i}" for i in range(len(self.agents))]
            
            x = np.arange(len(agent_ids))
            width = 0.35
            
            self.ax[0, 1].bar(x, self.metrics['path_lengths'])
            self.ax[0, 1].set_title('Agent Path Lengths')
            self.ax[0, 1].set_xticks(x)
            self.ax[0, 1].set_xticklabels(agent_ids)
            self.ax[0, 1].set_ylabel('Steps')
            
            # Plot agent idle times
            self.ax[1, 0].bar(x, self.metrics['idle_times'])
            self.ax[1, 0].set_title('Agent Idle Times')
            self.ax[1, 0].set_xticks(x)
            self.ax[1, 0].set_xticklabels(agent_ids)
            self.ax[1, 0].set_ylabel('Steps')
            
            # Plot rescue timeline
            if self.metrics['rescue_times']:
                timeline = np.zeros(self.step_count + 1)
                for rescue_time in self.metrics['rescue_times']:
                    if rescue_time <= self.step_count:
                        timeline[rescue_time] += 1
                
                cumulative = np.cumsum(timeline)
                self.ax[1, 1].plot(cumulative)
                self.ax[1, 1].set_title('Cumulative Rescues')
                self.ax[1, 1].set_xlabel('Time Steps')
                self.ax[1, 1].set_ylabel('Victims Rescued')
                self.ax[1, 1].grid(True)
            
            self.fig.tight_layout()
            self.canvas_fig.draw()
    
    def run_batch_trials(self):
        """Run multiple trials automatically and aggregate results"""
        # Disable UI during batch trials
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.run_trials_button.config(state=tk.DISABLED)
        
        # Parameters to vary
        agent_counts = [3, 5, 10]
        num_trials = 5
        strategy = self.strategy_var.get() if hasattr(self, 'strategy_var') else self.current_strategy
        
        # Results storage
        batch_results = {}
        
        # Run trials
        for agent_count in agent_counts:
            batch_results[agent_count] = {
                'completion_times': [],
                'coverage': [],
                'avg_path_length': [],
                'rescue_rates': [],
                'idle_times': []
            }
            
            for trial in range(num_trials):
                # Setup trial
                self.reset_simulation()
                if hasattr(self, 'agent_count_var'):
                    self.agent_count_var.set(agent_count)
                self.create_agents(agent_count, strategy)
                
                # Run simulation automatically until completion
                self.running = True
                self.step_count = 0
                
                # Run the simulation step by step without animation
                while self.running and self.step_count < self.max_steps:
                    prev_rescued = self.environment.victims_rescued
                    
                    # Update all agents
                    for agent in self.agents:
                        agent.update(self.agents)
                    
                    # Check if new victims were rescued
                    if self.environment.victims_rescued > prev_rescued:
                        self.metrics['rescue_times'].append(self.step_count)
                    
                    # Record rescue rate
                    self.metrics['rescue_rate'].append(
                        self.environment.victims_rescued / max(1, self.environment.victims_total)
                    )
                    
                    # Increment step count
                    self.step_count += 1
                    
                    # Check completion
                    if self.environment.is_mission_complete() or self.step_count >= self.max_steps:
                        self.running = False
                
                # Record metrics after trial
                self.record_final_metrics()
                
                # Store results for this trial
                batch_results[agent_count]['completion_times'].append(self.metrics['completion_time'])
                batch_results[agent_count]['coverage'].append(self.metrics['coverage'])
                batch_results[agent_count]['avg_path_length'].append(np.mean(self.metrics['path_lengths']))
                batch_results[agent_count]['rescue_rates'].append(self.metrics['rescue_rate'])
                batch_results[agent_count]['idle_times'].append(np.mean(self.metrics['idle_times']))
                
                print(f"Completed trial {trial+1}/{num_trials} for {agent_count} agents")
        
        # Save batch results
        self.save_batch_results(batch_results, strategy)
        
        # Display aggregated results
        self.display_batch_results(batch_results)
        
        # Re-enable UI
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.NORMAL, text="Start", command=self.start_simulation)
            self.reset_button.config(state=tk.NORMAL)
            self.run_trials_button.config(state=tk.NORMAL)
    
    def save_batch_results(self, results, strategy):
        """Save batch experiment results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_directory}/{strategy}_batch_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Strategy', 'Agent_Count', 'Trial', 'Completion_Time', 
                           'Coverage', 'Avg_Path_Length', 'Avg_Idle_Time'])
            
            for agent_count, data in results.items():
                for i in range(len(data['completion_times'])):
                    writer.writerow([
                        strategy,
                        agent_count,
                        i+1,
                        data['completion_times'][i],
                        data['coverage'][i],
                        data['avg_path_length'][i],
                        data['idle_times'][i]
                    ])
        
        print(f"Results saved to {filename}")
        return filename
    
    def display_batch_results(self, results):
        """Display aggregated results from batch trials"""
        # Show results window
        if hasattr(self, 'results_window'):
            self.results_window.deiconify()
            
            # Clear previous plots
            for ax_row in self.ax:
                for ax in ax_row:
                    ax.clear()
            
            # Plot average completion time for different agent counts
            agent_counts = list(results.keys())
            agent_counts.sort()  # Ensure counts are in ascending order
            avg_completion_times = [np.mean(results[count]['completion_times']) for count in agent_counts]
            std_completion_times = [np.std(results[count]['completion_times']) for count in agent_counts]
            
            self.ax[0, 0].bar(range(len(agent_counts)), avg_completion_times, yerr=std_completion_times)
            self.ax[0, 0].set_title('Average Completion Time by Agent Count')
            self.ax[0, 0].set_xticks(range(len(agent_counts)))
            self.ax[0, 0].set_xticklabels(agent_counts)
            self.ax[0, 0].set_xlabel('Number of Agents')
            self.ax[0, 0].set_ylabel('Time Steps')
            self.ax[0, 0].grid(True)
            
            # Plot average coverage
            avg_coverage = [np.mean(results[count]['coverage'])*100 for count in agent_counts]
            std_coverage = [np.std(results[count]['coverage'])*100 for count in agent_counts]
            
            self.ax[0, 1].bar(range(len(agent_counts)), avg_coverage, yerr=std_coverage)
            self.ax[0, 1].set_title('Average Coverage by Agent Count')
            self.ax[0, 1].set_xticks(range(len(agent_counts)))
            self.ax[0, 1].set_xticklabels(agent_counts)
            self.ax[0, 1].set_xlabel('Number of Agents')
            self.ax[0, 1].set_ylabel('Coverage (%)')
            self.ax[0, 1].grid(True)
            
            # Plot average path length
            avg_path_length = [np.mean(results[count]['avg_path_length']) for count in agent_counts]
            std_path_length = [np.std(results[count]['avg_path_length']) for count in agent_counts]
            
            self.ax[1, 0].bar(range(len(agent_counts)), avg_path_length, yerr=std_path_length)
            self.ax[1, 0].set_title('Average Path Length by Agent Count')
            self.ax[1, 0].set_xticks(range(len(agent_counts)))
            self.ax[1, 0].set_xticklabels(agent_counts)
            self.ax[1, 0].set_xlabel('Number of Agents')
            self.ax[1, 0].set_ylabel('Path Length')
            self.ax[1, 0].grid(True)
            
            # Plot average rescue rate over time for different agent counts
            for i, count in enumerate(agent_counts):
                # Get average rescue rate across trials at each time step
                rates = results[count]['rescue_rates']
                max_len = max(len(rate) for rate in rates)
                
                # Pad shorter arrays with final value
                padded_rates = []
                for rate in rates:
                    padded = rate.copy()
                    if len(padded) < max_len:
                        padded.extend([padded[-1]] * (max_len - len(padded)))
                    padded_rates.append(padded)
                
                # Calculate average rescue rate at each time step
                avg_rate = np.mean(padded_rates, axis=0)
                
                # Plot
                self.ax[1, 1].plot(avg_rate, label=f"{count} Agents")
            
            self.ax[1, 1].set_title('Average Rescue Rate Over Time')
            self.ax[1, 1].set_xlabel('Time Steps')
            self.ax[1, 1].set_ylabel('Fraction Rescued')
            self.ax[1, 1].legend()
            self.ax[1, 1].grid(True)
            
            self.fig.tight_layout()
            self.canvas_fig.draw()
        
        # Print summary
        print("\nBATCH EXPERIMENT SUMMARY")
        print("========================")
        print(f"Strategy: {self.strategy_var.get() if hasattr(self, 'strategy_var') else self.current_strategy}")
        for count in agent_counts:
            print(f"\nAgent Count: {count}")
            print(f"Average Completion Time: {np.mean(results[count]['completion_times']):.1f} ± {np.std(results[count]['completion_times']):.1f}")
            print(f"Average Coverage: {np.mean(results[count]['coverage'])*100:.1f}% ± {np.std(results[count]['coverage'])*100:.1f}%")
            print(f"Average Path Length: {np.mean(results[count]['avg_path_length']):.1f} ± {np.std(results[count]['avg_path_length']):.1f}")
            print(f"Average Idle Time: {np.mean(results[count]['idle_times']):.1f} ± {np.std(results[count]['idle_times']):.1f}")
    
    def compare_strategies(self):
        """Run experiments to compare different coordination strategies"""
        # Disable UI during comparison
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.run_trials_button.config(state=tk.DISABLED)
            self.compare_strategies_button.config(state=tk.DISABLED)
        
        # Parameters
        strategies = ["basic", "stigmergy", "communication"]
        agent_counts = [3, 5, 10]
        trials_per_config = 3
        
        # Results storage
        comparison_results = {}
        
        for strategy in strategies:
            comparison_results[strategy] = {}
            
            for agent_count in agent_counts:
                comparison_results[strategy][agent_count] = {
                    'completion_times': [],
                    'coverage': [],
                    'avg_path_length': [],
                    'idle_times': []
                }
                
                for trial in range(trials_per_config):
                    # Setup trial
                    self.reset_simulation()
                    if hasattr(self, 'agent_count_var'):
                        self.agent_count_var.set(agent_count)
                    self.create_agents(agent_count, strategy)
                    
                    # Run simulation automatically until completion
                    self.running = True
                    self.step_count = 0
                    
                    # Run the simulation step by step without animation
                    while self.running and self.step_count < self.max_steps:
                        prev_rescued = self.environment.victims_rescued
                        
                        # Update all agents
                        for agent in self.agents:
                            agent.update(self.agents)
                        
                        # Check if new victims were rescued
                        if self.environment.victims_rescued > prev_rescued:
                            self.metrics['rescue_times'].append(self.step_count)
                        
                        # Record rescue rate
                        self.metrics['rescue_rate'].append(
                            self.environment.victims_rescued / max(1, self.environment.victims_total)
                        )
                        
                        # Increment step count
                        self.step_count += 1
                        
                        # Check completion
                        if self.environment.is_mission_complete() or self.step_count >= self.max_steps:
                            self.running = False
                    
                    # Record metrics after trial
                    self.record_final_metrics()
                    
                    # Store results for this trial
                    comparison_results[strategy][agent_count]['completion_times'].append(self.metrics['completion_time'])
                    comparison_results[strategy][agent_count]['coverage'].append(self.metrics['coverage'])
                    comparison_results[strategy][agent_count]['avg_path_length'].append(np.mean(self.metrics['path_lengths']))
                    comparison_results[strategy][agent_count]['idle_times'].append(np.mean(self.metrics['idle_times']))
                    
                    print(f"Completed {strategy} trial {trial+1}/{trials_per_config} for {agent_count} agents")
        
        # Save comparison results
        self.save_comparison_results(comparison_results)
        
        # Display comparison results
        self.display_comparison_results(comparison_results)
        
        # Re-enable UI
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.NORMAL, text="Start", command=self.start_simulation)
            self.reset_button.config(state=tk.NORMAL)
            self.run_trials_button.config(state=tk.NORMAL)
            self.compare_strategies_button.config(state=tk.NORMAL)
    
    def save_comparison_results(self, results):
        """Save strategy comparison results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_directory}/strategy_comparison_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Strategy', 'Agent_Count', 'Trial', 'Completion_Time', 
                           'Coverage', 'Avg_Path_Length', 'Avg_Idle_Time'])
            
            for strategy, strategy_data in results.items():
                for agent_count, data in strategy_data.items():
                    for i in range(len(data['completion_times'])):
                        writer.writerow([
                            strategy,
                            agent_count,
                            i+1,
                            data['completion_times'][i],
                            data['coverage'][i],
                            data['avg_path_length'][i],
                            data['idle_times'][i]
                        ])
        
        print(f"Comparison results saved to {filename}")
        return filename
    
    def display_comparison_results(self, results):
        """Display results comparing different coordination strategies"""
        # Show results window
        if hasattr(self, 'results_window'):
            self.results_window.deiconify()
            
            # Clear previous plots
            for ax_row in self.ax:
                for ax in ax_row:
                    ax.clear()
            
            # Extract data for plotting
            strategies = list(results.keys())
            agent_counts = list(results[strategies[0]].keys())
            agent_counts.sort()
            
            # Set up x-axis positions for grouped bars
            x = np.arange(len(agent_counts))
            width = 0.25  
            
            # Plot completion times
            for i, strategy in enumerate(strategies):
                avg_times = [np.mean(results[strategy][count]['completion_times']) for count in agent_counts]
                std_times = [np.std(results[strategy][count]['completion_times']) for count in agent_counts]
                
                self.ax[0, 0].bar(x + (i-1)*width, avg_times, width, yerr=std_times, label=strategy.capitalize())
            
            self.ax[0, 0].set_title('Completion Time by Strategy')
            self.ax[0, 0].set_xticks(x)
            self.ax[0, 0].set_xticklabels(agent_counts)
            self.ax[0, 0].set_xlabel('Number of Agents')
            self.ax[0, 0].set_ylabel('Time Steps')
            self.ax[0, 0].legend()
            self.ax[0, 0].grid(True)
            
            # Plot coverage
            for i, strategy in enumerate(strategies):
                avg_coverage = [np.mean(results[strategy][count]['coverage'])*100 for count in agent_counts]
                std_coverage = [np.std(results[strategy][count]['coverage'])*100 for count in agent_counts]
                
                self.ax[0, 1].bar(x + (i-1)*width, avg_coverage, width, yerr=std_coverage, label=strategy.capitalize())
            
            self.ax[0, 1].set_title('Coverage by Strategy')
            self.ax[0, 1].set_xticks(x)
            self.ax[0, 1].set_xticklabels(agent_counts)
            self.ax[0, 1].set_xlabel('Number of Agents')
            self.ax[0, 1].set_ylabel('Coverage (%)')
            self.ax[0, 1].legend()
            self.ax[0, 1].grid(True)
            
            # Plot average path length
            for i, strategy in enumerate(strategies):
                avg_path = [np.mean(results[strategy][count]['avg_path_length']) for count in agent_counts]
                std_path = [np.std(results[strategy][count]['avg_path_length']) for count in agent_counts]
                
                self.ax[1, 0].bar(x + (i-1)*width, avg_path, width, yerr=std_path, label=strategy.capitalize())
            
            self.ax[1, 0].set_title('Path Length by Strategy')
            self.ax[1, 0].set_xticks(x)
            self.ax[1, 0].set_xticklabels(agent_counts)
            self.ax[1, 0].set_xlabel('Number of Agents')
            self.ax[1, 0].set_ylabel('Average Path Length')
            self.ax[1, 0].legend()
            self.ax[1, 0].grid(True)
            
            # Plot average idle time
            for i, strategy in enumerate(strategies):
                avg_idle = [np.mean(results[strategy][count]['idle_times']) for count in agent_counts]
                std_idle = [np.std(results[strategy][count]['idle_times']) for count in agent_counts]
                
                self.ax[1, 1].bar(x + (i-1)*width, avg_idle, width, yerr=std_idle, label=strategy.capitalize())
            
            self.ax[1, 1].set_title('Idle Time by Strategy')
            self.ax[1, 1].set_xticks(x)
            self.ax[1, 1].set_xticklabels(agent_counts)
            self.ax[1, 1].set_xlabel('Number of Agents')
            self.ax[1, 1].set_ylabel('Average Idle Time')
            self.ax[1, 1].legend()
            self.ax[1, 1].grid(True)
            
            self.fig.tight_layout()
            self.canvas_fig.draw()
        
        # Print statistical summary
        print("\nSTRATEGY COMPARISON SUMMARY")
        print("===========================")
        for strategy in strategies:
            print(f"\nStrategy: {strategy.upper()}")
            for count in agent_counts:
                print(f"  Agent Count: {count}")
                print(f"  Completion Time: {np.mean(results[strategy][count]['completion_times']):.1f} ± {np.std(results[strategy][count]['completion_times']):.1f}")
                print(f"  Coverage: {np.mean(results[strategy][count]['coverage'])*100:.1f}% ± {np.std(results[strategy][count]['coverage'])*100:.1f}%")
    
    def export_results(self):
        """Export the current results to a CSV file"""
        if not self.metrics['completion_time']:
            print("No results to export. Run a simulation first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_directory}/single_run_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Strategy', 'Agent_Count', 'Completion_Time', 'Coverage', 
                           'Victims_Total', 'Victims_Rescued'])
            
            writer.writerow([
                self.current_strategy,
                len(self.agents),
                self.metrics['completion_time'],
                self.metrics['coverage'],
                self.environment.victims_total,
                self.environment.victims_rescued
            ])
            
            # Write agent-specific data
            writer.writerow([])
            writer.writerow(['Agent_ID', 'Path_Length', 'Idle_Time', 'Victims_Rescued'])
            
            for i, agent in enumerate(self.agents):
                writer.writerow([
                    f"Agent{i}",
                    self.metrics['path_lengths'][i] if i < len(self.metrics['path_lengths']) else 0,
                    self.metrics['idle_times'][i] if i < len(self.metrics['idle_times']) else 0,
                    agent.rescued_count
                ])
            
            # Write rescue timeline
            writer.writerow([])
            writer.writerow(['Step', 'Cumulative_Rescues'])
            
            timeline = np.zeros(self.step_count + 1)
            for rescue_time in self.metrics['rescue_times']:
                if rescue_time <= self.step_count:
                    timeline[rescue_time] += 1
            
            cumulative = np.cumsum(timeline)
            for step in range(self.step_count + 1):
                writer.writerow([step, cumulative[step]])
        
        print(f"Results exported to {filename}")
    
    def draw(self):
        """Draw the environment and all agents"""
        self.environment.draw(self.canvas)
        for agent in self.agents:
            agent.draw(self.canvas)
        
        # Display current step count
        self.canvas.delete("step_count")
        self.canvas.create_text(
            self.environment.width - 100,
            self.environment.height + 20,
            text=f"Steps: {self.step_count}",
            tags="step_count"
        )
    
    # New methods for automated experimentation
    
    def run_automated_phase(self, phase):
        """Run a specific experimental phase"""
        if phase == 1:
            self.run_phase1()
        elif phase == 2:
            self.run_phase2()
        elif phase == 3:
            self.run_phase3()
        else:
            print(f"Unknown phase: {phase}")
    
    def run_phase1(self, repetitions=5):
        """
        Run Phase 1: Agent Count and Strategy Comparison
        Tests all combinations of strategies and agent counts.
        """
        strategies = ["basic", "stigmergy", "communication"]
        agent_counts = [3, 5, 10]
        
        # Fixed settings for Phase 1
        obstacle_density = 0.15
        victim_count = 10
        
        # Disable UI if in GUI mode
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.run_trials_button.config(state=tk.DISABLED)
        
        print("\n===== PHASE 1: AGENT COUNT AND STRATEGY COMPARISON =====")
        print(f"Testing {len(strategies)} strategies × {len(agent_counts)} agent counts × {repetitions} repetitions")
        print(f"Fixed settings: obstacle_density={obstacle_density}, victims={victim_count}\n")
        
        # Create timestamp for this batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_directory}/phase1_results_{timestamp}.csv"
        
        # Create CSV file and write header
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Phase', 'Strategy', 'Agent_Count', 'Obstacle_Density', 
                'Victim_Count', 'Trial', 'Completion_Time', 'Coverage', 
                'Avg_Path_Length', 'Avg_Idle_Time', 'Victims_Rescued', 
                'Victims_Total', 'Success_Rate'
            ])
        
        # Total trial count for progress tracking
        total_trials = len(strategies) * len(agent_counts) * repetitions
        trial_counter = 0
        
        # Run trials
        for strategy in strategies:
            for agent_count in agent_counts:
                for rep in range(repetitions):
                    trial_counter += 1
                    print(f"Trial {trial_counter}/{total_trials}: {strategy} strategy with {agent_count} agents (rep {rep+1}/{repetitions})")
                    
                    # Set seed for reproducibility while maintaining variety between trials
                    trial_seed = (hash(strategy) + agent_count * 100 + rep) % 10000
                    random.seed(trial_seed)
                    np.random.seed(trial_seed)
                    
                    # Reset simulation
                    self.reset_simulation()
                    
                    # Set up environment with fixed settings
                    if hasattr(self, 'obstacle_density_var'):
                        self.obstacle_density_var.set(obstacle_density)
                    if hasattr(self, 'victim_count_var'):
                        self.victim_count_var.set(victim_count)
                    
                    self.setup_environment()
                    
                    # Create agents based on strategy
                    self.create_agents(agent_count, strategy)
                    
                    # Run simulation until completion
                    self.running = True
                    self.step_count = 0
                    self.environment.start_mission()
                    
                    # Run without animation
                    while self.running and self.step_count < self.max_steps:
                        prev_rescued = self.environment.victims_rescued
                        
                        # Update all agents
                        for agent in self.agents:
                            agent.update(self.agents)
                        
                        # Check if new victims were rescued
                        if self.environment.victims_rescued > prev_rescued:
                            self.metrics['rescue_times'].append(self.step_count)
                        
                        # Record rescue rate
                        self.metrics['rescue_rate'].append(
                            self.environment.victims_rescued / max(1, self.environment.victims_total)
                        )
                        
                        # Increment step count
                        self.step_count += 1
                        
                        # Check if mission is complete
                        if self.environment.is_mission_complete() or self.step_count >= self.max_steps:
                            self.running = False
                    
                    # Record metrics
                    self.record_final_metrics()
                    
                    # Calculate success rate
                    success_rate = self.environment.victims_rescued / max(1, self.environment.victims_total)
                    
                    # Append to CSV
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            1,  # Phase
                            strategy,
                            agent_count,
                            obstacle_density,
                            victim_count,
                            rep + 1,
                            self.metrics['completion_time'],
                            self.metrics['coverage'],
                            np.mean(self.metrics['path_lengths']),
                            np.mean(self.metrics['idle_times']),
                            self.environment.victims_rescued,
                            self.environment.victims_total,
                            success_rate
                        ])
        
        # Re-enable UI if in GUI mode
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.NORMAL, text="Start", command=self.start_simulation)
            self.reset_button.config(state=tk.NORMAL)
            self.run_trials_button.config(state=tk.NORMAL)
        
        print(f"\nPhase 1 complete. Results saved to {filename}")
        return filename
    
    def run_phase2(self, repetitions=3):
        """
        Run Phase 2: Environmental Impact
        Tests how environment complexity affects performance.
        """
        strategies = ["basic", "stigmergy", "communication"]
        obstacle_densities = [0.05, 0.15, 0.25]  # Low, medium, high complexity
        
        # Fixed settings for Phase 2
        agent_count = 5  # Middle value
        victim_count = 10
        
        # Disable UI if in GUI mode
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.run_trials_button.config(state=tk.DISABLED)
        
        print("\n===== PHASE 2: ENVIRONMENTAL IMPACT =====")
        print(f"Testing {len(strategies)} strategies × {len(obstacle_densities)} complexity levels × {repetitions} repetitions")
        print(f"Fixed settings: agent_count={agent_count}, victims={victim_count}\n")
        
        # Create timestamp for this batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_directory}/phase2_results_{timestamp}.csv"
        
        # Create CSV file and write header
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Phase', 'Strategy', 'Agent_Count', 'Obstacle_Density', 
                'Victim_Count', 'Trial', 'Completion_Time', 'Coverage', 
                'Avg_Path_Length', 'Avg_Idle_Time', 'Victims_Rescued', 
                'Victims_Total', 'Success_Rate'
            ])
        
        # Total trial count for progress tracking
        total_trials = len(strategies) * len(obstacle_densities) * repetitions
        trial_counter = 0
        
        # Run trials
        for strategy in strategies:
            for obstacle_density in obstacle_densities:
                for rep in range(repetitions):
                    trial_counter += 1
                    print(f"Trial {trial_counter}/{total_trials}: {strategy} strategy with {obstacle_density} obstacle density (rep {rep+1}/{repetitions})")
                    
                    # Set seed for reproducibility
                    trial_seed = (hash(strategy) + int(obstacle_density * 1000) + rep) % 10000
                    random.seed(trial_seed)
                    np.random.seed(trial_seed)
                    
                    # Reset simulation
                    self.reset_simulation()
                    
                    # Set up environment with specified obstacle density
                    if hasattr(self, 'obstacle_density_var'):
                        self.obstacle_density_var.set(obstacle_density)
                    if hasattr(self, 'victim_count_var'):
                        self.victim_count_var.set(victim_count)
                    
                    self.setup_environment()
                    
                    # Create agents based on strategy
                    self.create_agents(agent_count, strategy)
                    
                    # Run simulation until completion
                    self.running = True
                    self.step_count = 0
                    self.environment.start_mission()
                    
                    # Run without animation
                    while self.running and self.step_count < self.max_steps:
                        prev_rescued = self.environment.victims_rescued
                        
                        # Update all agents
                        for agent in self.agents:
                            agent.update(self.agents)
                        
                        # Check if new victims were rescued
                        if self.environment.victims_rescued > prev_rescued:
                            self.metrics['rescue_times'].append(self.step_count)
                        
                        # Record rescue rate
                        self.metrics['rescue_rate'].append(
                            self.environment.victims_rescued / max(1, self.environment.victims_total)
                        )
                        
                        # Increment step count
                        self.step_count += 1
                        
                        # Check if mission is complete
                        if self.environment.is_mission_complete() or self.step_count >= self.max_steps:
                            self.running = False
                    
                    # Record metrics
                    self.record_final_metrics()
                    
                    # Calculate success rate
                    success_rate = self.environment.victims_rescued / max(1, self.environment.victims_total)
                    
                    # Append to CSV
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            2,  # Phase
                            strategy,
                            agent_count,
                            obstacle_density,
                            victim_count,
                            rep + 1,
                            self.metrics['completion_time'],
                            self.metrics['coverage'],
                            np.mean(self.metrics['path_lengths']),
                            np.mean(self.metrics['idle_times']),
                            self.environment.victims_rescued,
                            self.environment.victims_total,
                            success_rate
                        ])
        
        # Re-enable UI if in GUI mode
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.NORMAL, text="Start", command=self.start_simulation)
            self.reset_button.config(state=tk.NORMAL)
            self.run_trials_button.config(state=tk.NORMAL)
        
        print(f"\nPhase 2 complete. Results saved to {filename}")
        return filename
    
    def run_phase3(self, repetitions=1):
        """
        Run Phase 3: Communication Constraints
        Tests how communication range affects performance of communicating agents.
        """
        agent_counts = [3, 5, 10]
        comm_ranges = [float('inf'), 5, 2]  # Unlimited, medium, limited
        
        # Fixed settings for Phase 3
        strategy = "communication"  # Only test communication-based agents
        obstacle_density = 0.15  # Medium complexity
        victim_count = 10
        
        # Disable UI if in GUI mode
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.run_trials_button.config(state=tk.DISABLED)
        
        print("\n===== PHASE 3: COMMUNICATION CONSTRAINTS =====")
        print(f"Testing {len(agent_counts)} agent counts × {len(comm_ranges)} communication ranges × {repetitions} repetitions")
        print(f"Fixed settings: strategy={strategy}, obstacle_density={obstacle_density}, victims={victim_count}\n")
        
        # Create timestamp for this batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_directory}/phase3_results_{timestamp}.csv"
        
        # Create CSV file and write header
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Phase', 'Strategy', 'Agent_Count', 'Obstacle_Density', 
                'Victim_Count', 'Comm_Range', 'Trial', 'Completion_Time', 
                'Coverage', 'Avg_Path_Length', 'Avg_Idle_Time', 
                'Victims_Rescued', 'Victims_Total', 'Success_Rate'
            ])
        
        # Total trial count for progress tracking
        total_trials = len(agent_counts) * len(comm_ranges) * repetitions
        trial_counter = 0
        
        # Run trials
        for agent_count in agent_counts:
            for comm_range in comm_ranges:
                for rep in range(repetitions):
                    trial_counter += 1
                    range_label = "unlimited" if comm_range == float('inf') else str(comm_range)
                    print(f"Trial {trial_counter}/{total_trials}: {agent_count} agents with {range_label} comm range (rep {rep+1}/{repetitions})")
                    
                    # Set seed for reproducibility
                    trial_seed = (agent_count * 100 + (999 if comm_range == float('inf') else int(comm_range * 100)) + rep) % 10000
                    random.seed(trial_seed)
                    np.random.seed(trial_seed)
                    
                    # Reset simulation
                    self.reset_simulation()
                    
                    # Set up environment with fixed settings
                    if hasattr(self, 'obstacle_density_var'):
                        self.obstacle_density_var.set(obstacle_density)
                    if hasattr(self, 'victim_count_var'):
                        self.victim_count_var.set(victim_count)
                    
                    self.setup_environment()
                    
                    # Create communication agents with specified comm range
                    self.create_agents(agent_count, strategy, comm_range)
                    
                    # Run simulation until completion
                    self.running = True
                    self.step_count = 0
                    self.environment.start_mission()
                    
                    # Run without animation
                    while self.running and self.step_count < self.max_steps:
                        prev_rescued = self.environment.victims_rescued
                        
                        # Update all agents
                        for agent in self.agents:
                            agent.update(self.agents)
                        
                        # Check if new victims were rescued
                        if self.environment.victims_rescued > prev_rescued:
                            self.metrics['rescue_times'].append(self.step_count)
                        
                        # Record rescue rate
                        self.metrics['rescue_rate'].append(
                            self.environment.victims_rescued / max(1, self.environment.victims_total)
                        )
                        
                        # Increment step count
                        self.step_count += 1
                        
                        # Check if mission is complete
                        if self.environment.is_mission_complete() or self.step_count >= self.max_steps:
                            self.running = False
                    
                    # Record metrics
                    self.record_final_metrics()
                    
                    # Calculate success rate
                    success_rate = self.environment.victims_rescued / max(1, self.environment.victims_total)
                    
                    # Append to CSV
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            3,  # Phase
                            strategy,
                            agent_count,
                            obstacle_density,
                            victim_count,
                            "inf" if comm_range == float('inf') else comm_range,
                            rep + 1,
                            self.metrics['completion_time'],
                            self.metrics['coverage'],
                            np.mean(self.metrics['path_lengths']),
                            np.mean(self.metrics['idle_times']),
                            self.environment.victims_rescued,
                            self.environment.victims_total,
                            success_rate
                        ])
        
        # Re-enable UI if in GUI mode
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.NORMAL, text="Start", command=self.start_simulation)
            self.reset_button.config(state=tk.NORMAL)
            self.run_trials_button.config(state=tk.NORMAL)
        
        print(f"\nPhase 3 complete. Results saved to {filename}")
        return filename