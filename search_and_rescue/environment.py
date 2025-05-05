"""
Environment module for Search and Rescue simulation.
Contains the disaster environment implementation.
"""

import tkinter as tk
import random
import math
import numpy as np
import time

class DisasterEnvironment:
    """Simple 2D grid disaster environment for search and rescue simulation"""
    
    def __init__(self, grid_size=20, cell_size=30):
        self.grid_size = grid_size  # 20x20 grid
        self.cell_size = cell_size  # Each cell is 30x30 pixels
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size
        
        # Grid representation: 0=empty, 1=obstacle, 2=victim
        self.grid = np.zeros((grid_size, grid_size))
        
        # Metrics for experiment tracking
        self.victims_total = 0
        self.victims_rescued = 0
        self.start_time = None
        
    def create_obstacles(self, obstacle_density=0.1):
        """Add random obstacles to the environment based on density"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < obstacle_density:
                    self.grid[i][j] = 1  # 1 represents obstacle
    
    def create_disaster_zone(self, victim_count=10):
        """Create a disaster zone with victims clustered in certain areas"""
        # Create 1-3 disaster zones (clusters of victims)
        zone_count = random.randint(1, 3)
        
        for _ in range(zone_count):
            # Choose a center for the disaster zone
            center_x = random.randint(3, self.grid_size - 4)
            center_y = random.randint(3, self.grid_size - 4)
            
            # Determine zone size
            zone_radius = random.randint(2, 4)
            
            # Place victims in the zone
            victims_per_zone = victim_count // zone_count
            placed = 0
            
            while placed < victims_per_zone:
                # Get location near the center of the zone
                offset_x = random.randint(-zone_radius, zone_radius)
                offset_y = random.randint(-zone_radius, zone_radius)
                
                x = center_x + offset_x
                y = center_y + offset_y
                
                # Ensure the location is within grid and not on an obstacle
                if (0 <= x < self.grid_size and 0 <= y < self.grid_size and 
                    self.grid[x][y] == 0):
                    self.grid[x][y] = 2  # 2 represents victim
                    placed += 1
                    self.victims_total += 1
    
    def get_cell_type(self, x, y):
        """Return the type of cell at grid coordinates (x, y)"""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.grid[x][y]
        return -1  # Out of bounds
    
    def is_valid_position(self, x, y):
        """Check if a position is valid (within bounds and not an obstacle)"""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.grid[x][y] != 1  # Not an obstacle
        return False
    
    def rescue_victim(self, x, y):
        """Attempt to rescue a victim at location (x, y)"""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if self.grid[x][y] == 2:  # If there's a victim
                self.grid[x][y] = 0   # Clear cell
                self.victims_rescued += 1
                return True
        return False
    
    def draw(self, canvas):
        """Draw the environment on a tkinter canvas"""
        # Clear canvas
        canvas.delete("environment")
        
        # Draw grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Choose color based on cell type
                if self.grid[i][j] == 0:    # Empty
                    color = "white"
                elif self.grid[i][j] == 1:  # Obstacle
                    color = "gray"
                elif self.grid[i][j] == 2:  # Victim
                    color = "red"
                
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black", tags="environment")
        
        # Draw grid lines for better visibility
        for i in range(self.grid_size + 1):
            x = i * self.cell_size
            canvas.create_line(x, 0, x, self.height, fill="black", tags="environment")
            canvas.create_line(0, x, self.width, x, fill="black", tags="environment")
        
        # Display metrics
        canvas.delete("metrics")
        canvas.create_text(
            self.width // 2, 
            self.height + 20, 
            text=f"Victims: {self.victims_rescued}/{self.victims_total}", 
            tags="metrics"
        )
        
        # Display elapsed time if mission has started
        if self.start_time:
            elapsed = time.time() - self.start_time
            canvas.create_text(
                self.width // 2,
                self.height + 40,
                text=f"Time: {elapsed:.1f} seconds",
                tags="metrics"
            )
    
    def start_mission(self):
        """Start the search and rescue mission timer"""
        self.start_time = time.time()
    
    def is_mission_complete(self):
        """Check if all victims have been rescued"""
        return self.victims_rescued == self.victims_total and self.victims_total > 0