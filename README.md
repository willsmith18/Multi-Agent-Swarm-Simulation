# Multi-Agent Search and Rescue Simulation

This project investigates how different coordination strategies affect multi-agent search and rescue efficiency under varying environmental conditions.

## Overview

The simulation features autonomous agents operating in a 2D grid-based disaster environment with obstacles and victims. The agents' goal is to locate and rescue all victims as efficiently as possible. The project examines how different coordination approaches affect team performance.

## Features

- **Environment**: A configurable 2D grid with obstacles and clustered victims
- **Three Coordination Strategies**:
  - **Basic**: Simple rule-based behavior with minimal coordination
  - **Stigmergy**: Indirect coordination through environment markers (pheromone-based)
  - **Direct Communication**: Explicit message passing between agents
- **Experimental Framework**: Automated experiments with varying agent counts and environmental conditions
- **Metrics**: Rescue completion time, area coverage, path length, and coordination overhead
- **Visualization**: Real-time display of agent behavior and performance metrics

## Project Structure

```
search_and_rescue/
├── environment.py       # Environment implementation
├── agents.py            # Agent implementations
├── controller.py        # Experiment controller
├── main.py              # Entry point
└── README.md            # Documentation
```

## Usage

### Running the Simulation with GUI

```bash
python main.py
```

Optional arguments:
- `--strategy`: Coordination strategy (basic, stigmergy, communication)
- `--agents`: Number of agents (default: 3)
- `--grid_size`: Size of grid (default: 20)
- `--obstacle_density`: Density of obstacles (default: 0.15)
- `--victims`: Number of victims (default: 10)

Example:
```bash
python main.py --strategy stigmergy --agents 5
```

### Running Batch Experiments

```bash
python main.py --batch --strategy basic --agents 5 --trials 10
```

This will run experiments without the GUI and save results to CSV files.

## Coordination Strategies

### Basic Coordination
Simple rule-based behavior where agents prioritize:
1. Rescuing visible victims
2. Exploring unvisited cells
3. Random movement when no better option exists

### Stigmergy Coordination
Agents deposit virtual "pheromones" in visited cells and prefer to move to cells with lower pheromone levels, encouraging exploration of unexplored areas without direct communication.

### Direct Communication
Agents can communicate with other agents within range to share:
- Known victim locations
- Planned paths
- Assigned rescue targets

This enables efficient task allocation and path planning.

## Experimental Questions

The simulation is designed to investigate:

1. How does performance scale with increasing agent numbers (3, 5, 10)?
2. How do different coordination strategies compare when communication is limited?
3. Which approaches are most robust to sensor limitations and environmental uncertainty?

## Metrics

- **Completion Time**: Steps required to rescue all victims
- **Coverage**: Percentage of navigable environment explored
- **Path Length**: Total distance traveled by agents
- **Idle Time**: Steps where agents make no progress
- **Rescue Rate**: Victims rescued over time

## Adding New Features

### New Coordination Strategies
Extend the `Agent` class and implement a new coordination approach in the `update()` method.

### Environmental Variations
Modify the `DisasterEnvironment` class to add new features like dynamic obstacles or time-dependent scenarios.

### Additional Metrics
Expand the metrics tracking in the `ExperimentController` class.
