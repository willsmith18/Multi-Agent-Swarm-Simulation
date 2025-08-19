"""
Utility Functions for Search and Rescue Simulation

This module contains utility functions for data visualization, analysis, and reporting
of multi-agent search and rescue experiments.

Modules:
- visualization: Comprehensive analysis and plotting functions for experimental results

Functions available:
- Data processing and statistical analysis
- Publication-quality visualization generation
- Experimental result comparison and reporting
- Heatmap and statistical plot creation

Example Usage:
    from search_rescue.utils.visualization import create_comprehensive_report
    
    # Generate complete analysis report
    create_comprehensive_report(['phase1_results.csv'], 'output_dir')
"""

# Import main visualization functions for easier access
try:
    from .visualization import (
        create_comprehensive_report,
        create_heatmap,
        create_boxplot, 
        create_lineplot
    )
    
    __all__ = [
        'create_comprehensive_report',
        'create_heatmap',
        'create_boxplot',
        'create_lineplot'
    ]
    
except ImportError:
    # Handle case where visualization module might not be available
    # or dependencies are missing
    __all__ = []

# Utility constants
ANALYSIS_OUTPUT_FORMATS = ['png', 'pdf', 'svg']
DEFAULT_DPI = 300
DEFAULT_FIGURE_SIZE = (10, 8)