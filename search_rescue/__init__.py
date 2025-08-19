"""
Multi-Agent Search and Rescue Simulation Package

This package implements advanced coordination strategies for multi-agent search and rescue operations,
featuring three distinct coordination algorithms: basic rule-based, stigmergy-based, and 
communication-based coordination.

Key Features:
- Multi-agent coordination algorithms (basic, stigmergy, communication)
- A* pathfinding implementation for optimal route planning
- Comprehensive experimental framework with statistical analysis
- Real-time visualization and performance metrics
- Scalability testing across different team sizes and environments

Modules:
- agents: Agent implementations with different coordination strategies
- environment: Disaster environment simulation with obstacles and victims
- controller: Experiment management and GUI interface
- utils: Utility functions for visualization and analysis

Example Usage:
    from search_rescue.controller import ExperimentController
    from search_rescue.agents import CommunicatingAgent
    from search_rescue.environment import DisasterEnvironment
    
    # Create environment and agents
    env = DisasterEnvironment(grid_size=20)
    controller = ExperimentController()
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "William Smith"
__email__ = "wsmith4313@outlook.com"
__institution__ = "University of Nottingham"

# Import main classes for easier access
from .agents import Agent, StigmergyAgent, CommunicatingAgent
from .environment import DisasterEnvironment
from .controller import ExperimentController

# Define what gets imported with "from search_rescue import *"
__all__ = [
    'Agent',
    'StigmergyAgent', 
    'CommunicatingAgent',
    'DisasterEnvironment',
    'ExperimentController'
]

# Package-level constants
DEFAULT_GRID_SIZE = 20
DEFAULT_CELL_SIZE = 30
DEFAULT_MAX_STEPS = 1000

# Coordination strategies available
COORDINATION_STRATEGIES = ['basic', 'stigmergy', 'communication']

# Experimental phases
EXPERIMENTAL_PHASES = {
    1: "Agent Count and Strategy Comparison",
    2: "Environmental Impact Analysis", 
    3: "Communication Constraints Study"
}