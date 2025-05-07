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

    def verify_victims_accessibility(self):
        """
        Verify that all victims in the environment are accessible.
        Returns:
            tuple: (all_accessible, inaccessible_victims) where all_accessible is a boolean
                and inaccessible_victims is a list of (x, y) coordinates of unreachable victims
        """
        inaccessible_victims = []
        victim_positions = []
        
        # Find all victim positions
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 2:  # Victim
                    victim_positions.append((i, j))
        
        # For each victim, check if it's accessible from any valid cell
        for victim_pos in victim_positions:
            if not self._is_accessible(victim_pos):
                inaccessible_victims.append(victim_pos)
        
        return len(inaccessible_victims) == 0, inaccessible_victims

    def _is_accessible(self, target_pos):
        """
        Check if the target position is accessible from any valid starting position.
        Uses breadth-first search to find a path.
        
        Args:
            target_pos: (x, y) coordinates of the target position
            
        Returns:
            bool: True if target is accessible, False otherwise
        """
        # BFS to find a path to victim
        queue = []
        visited = set()
        
        # Add all empty cells as potential starting points
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 0:  # Empty cell
                    queue.append((i, j))
                    visited.add((i, j))
        
        # Cardinal directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while queue:
            current = queue.pop(0)
            
            # Check if we're adjacent to the target
            if (abs(current[0] - target_pos[0]) == 1 and current[1] == target_pos[1]) or \
            (abs(current[1] - target_pos[1]) == 1 and current[0] == target_pos[0]):
                return True
            
            # Try all four directions
            for dx, dy in directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                # Check if the next position is valid and not visited
                if (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size and
                    self.grid[next_x][next_y] != 1 and  # Not an obstacle
                    next_pos not in visited):
                    
                    queue.append(next_pos)
                    visited.add(next_pos)
        
        return False

    def create_disaster_zone(self, victim_count=10):
        """Create a disaster zone with victims clustered in certain areas, ensuring all victims are accessible"""
        # Create 1-3 disaster zones (clusters of victims)
        zone_count = random.randint(1, 3)
        max_attempts = 100  # Limit attempts to prevent infinite loops
        
        # Reset victim count
        self.victims_total = 0
        
        for _ in range(zone_count):
            # Choose a center for the disaster zone
            center_x = random.randint(3, self.grid_size - 4)
            center_y = random.randint(3, self.grid_size - 4)
            
            # Determine zone size
            zone_radius = random.randint(2, 4)
            
            # Place victims in the zone
            victims_per_zone = victim_count // zone_count
            placed = 0
            attempts = 0
            
            while placed < victims_per_zone and attempts < max_attempts:
                # Get location near the center of the zone
                offset_x = random.randint(-zone_radius, zone_radius)
                offset_y = random.randint(-zone_radius, zone_radius)
                
                x = center_x + offset_x
                y = center_y + offset_y
                
                # Ensure the location is within grid and not on an obstacle
                if (0 <= x < self.grid_size and 0 <= y < self.grid_size and 
                    self.grid[x][y] == 0):
                    
                    # Check if placing a victim here would have at least one adjacent empty cell
                    has_adjacent_empty = False
                    for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                            self.grid[nx][ny] == 0):
                            has_adjacent_empty = True
                            break
                    
                    if has_adjacent_empty:
                        self.grid[x][y] = 2  # Place victim
                        placed += 1
                        self.victims_total += 1
                
                attempts += 1
        
        # Verify all victims are accessible
        all_accessible, inaccessible = self.verify_victims_accessibility()
        if not all_accessible:
            print(f"Warning: {len(inaccessible)} victims are inaccessible! Removing them.")
            for pos in inaccessible:
                self.grid[pos[0]][pos[1]] = 0  # Remove inaccessible victim
                self.victims_total -= 1
    
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