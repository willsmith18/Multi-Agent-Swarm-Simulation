# Enhanced Multi-Agent Search and Rescue Simulation

This project implements a multi-agent coordination simulation for search and rescue scenarios, allowing comparison of different coordination strategies with automated experiment support.

## Features

- **Multiple Coordination Strategies**:
  - Basic: Simple rule-based behavior with minimal coordination
  - Stigmergy: Indirect coordination through environment markers
  - Communication: Direct inter-agent communication

- **Interactive GUI Mode**:
  - Visual simulation of agent behavior
  - Real-time performance metrics
  - Adjustable simulation parameters

- **Automated Experimentation**:
  - Pre-configured experiment phases
  - Batch processing support
  - Comprehensive data collection

- **Result Analysis**:
  - Statistical summaries and visualizations
  - Strategy comparison tools
  - Performance evaluation metrics

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required packages: numpy, matplotlib, pandas, seaborn, scipy, tqdm

Install required packages:
```bash
pip install numpy matplotlib pandas seaborn scipy tqdm
```

### Running the Simulation

#### GUI Mode

```bash
python main.py --mode gui --strategy stigmergy --agents 5
```

#### Experiment Modes

Run a specific experimental phase:
```bash
# Phase 1: Agent Count and Strategy Comparison
python main.py --mode phase1 --repetitions 3 --output experiment_results/phase1

# Phase 2: Environmental Impact
python main.py --mode phase2 --repetitions 2 --output experiment_results/phase2

# Phase 3: Communication Constraints
python main.py --mode phase3 --repetitions 1 --output experiment_results/phase3
```

Run a custom batch experiment:
```bash
python main.py --mode batch --strategy communication --agents 10 --repetitions 5
```

Analyze experiment results:
```bash
python main.py --mode analyze --files experiment_results/phase1_results_*.csv experiment_results/phase2_results_*.csv
```

## Experimental Phases

### Phase 1: Agent Count and Strategy Comparison
Tests all combinations of:
- 3 coordination strategies (basic, stigmergy, communication)
- 3 agent counts (3, 5, 10)
- Default: 5 repetitions per configuration

Fixed settings:
- Medium environmental complexity (0.15 obstacle density)
- 10 victims in clustered distribution
- Communication range of 5 (for communication strategy)

### Phase 2: Environmental Impact
Tests how environment affects performance:
- 3 coordination strategies
- 3 environmental complexities (0.05, 0.15, 0.25 obstacle density)
- Default: 3 repetitions per configuration

Fixed settings:
- 5 agents
- 10 victims in clustered distribution

### Phase 3: Communication Constraints
Tests communication-based agents only:
- 3 communication ranges (unlimited, 5 cells, 2 cells)
- 3 agent counts (3, 5, 10)
- Default: 1 repetition per configuration

Fixed settings:
- Medium environmental complexity
- 10 victims in clustered distribution


## Project Structure

```
search_and_rescue/
├── environment.py       # Environment implementation
├── agents.py            # Agent implementations
├── controller.py        # Experiment controller with GUI and batch capabilities
├── main.py              # Entry point with command-line options
├── visualization_utils.py # Optional visualization utilities
└── README.md            # This file
```

## Example Commands for Common Tasks

### Running All Experiment Phases

```bash
# Create a directory for all results
mkdir -p all_experiments

# Run all three phases with reduced repetitions for speed
python main.py --mode phase1 --repetitions 3 --output all_experiments/phase1
python main.py --mode phase2 --repetitions 2 --output all_experiments/phase2
python main.py --mode phase3 --repetitions 1 --output all_experiments/phase3

# Analyze all results together
python main.py --mode analyze --output all_experiments/analysis --files all_experiments/phase*/*.csv
```

### Comparative Analysis

```bash
# Run two strategies with same agent count for comparison
python main.py --mode batch --strategy basic --agents 5 --repetitions 5 --output compare/basic
python main.py --mode batch --strategy stigmergy --agents 5 --repetitions 5 --output compare/stigmergy

# Analyze the comparison
python main.py --mode analyze --output compare/analysis --files compare/basic/*.csv compare/stigmergy/*.csv
```

### Quick Test Run

```bash
# Single trial with visual feedback
python main.py --mode gui --strategy communication --agents 3
```