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


class Agent:
    """Search and rescue agent with basic capabilities"""
    
    def __init__(self, name, environment, x=0, y=0):
        self.name = name
        self.environment = environment
        
        # Starting position
        self.x = x
        self.y = y
        
        # Agent properties
        self.size = environment.cell_size * 0.8
        self.sensor_range = 2  # Can see 2 cells in each direction
        self.color = "blue"
        
        # Agent state
        self.rescued_count = 0
        self.path = []  # For coordination strategies to use
        
        # For implementing different coordination strategies
        self.coordination_type = "basic"  # Options: "basic", "stigmergy"
        self.pheromone_map = None  # For stigmergy coordination
        
    def place_at_random_position(self):
        """Place agent at a random valid position in the environment"""
        while True:
            x = random.randint(0, self.environment.grid_size - 1)
            y = random.randint(0, self.environment.grid_size - 1)
            if self.environment.is_valid_position(x, y):
                self.x = x
                self.y = y
                break
    
    def sense_environment(self):
        """Get information about nearby cells within sensor range"""
        sensed_data = []
        
        for dx in range(-self.sensor_range, self.sensor_range + 1):
            for dy in range(-self.sensor_range, self.sensor_range + 1):
                # Skip sensing own position
                if dx == 0 and dy == 0:
                    continue
                
                cell_x = self.x + dx
                cell_y = self.y + dy
                cell_type = self.environment.get_cell_type(cell_x, cell_y)
                
                if cell_type != -1:  # Not out of bounds
                    sensed_data.append({
                        'x': cell_x,
                        'y': cell_y,
                        'type': cell_type
                    })
        
        return sensed_data
    
    def basic_coordination(self, sensed_data):
        """Simple rule-based movement strategy"""
        # Check if there's a victim in sensed cells
        for cell in sensed_data:
            if cell['type'] == 2:  # Victim
                # Move toward victim
                dx = cell['x'] - self.x
                dy = cell['y'] - self.y
                
                # Get only one step closer
                move_x = 0 if dx == 0 else dx // abs(dx)
                move_y = 0 if dy == 0 else dy // abs(dy)
                
                # Try to move
                if self.environment.is_valid_position(self.x + move_x, self.y + move_y):
                    self.x += move_x
                    self.y += move_y
                    return
        
        # If no victim is found, move randomly
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            if self.environment.is_valid_position(self.x + dx, self.y + dy):
                self.x += dx
                self.y += dy
                return
    
    def stigmergy_coordination(self, sensed_data, agents):
        """Stigmergy-based coordination using virtual pheromones"""
        # Initialize pheromone map if none exists
        if self.pheromone_map is None:
            self.pheromone_map = np.zeros((self.environment.grid_size, self.environment.grid_size))
        
        # Leave pheromone in current location
        self.pheromone_map[self.x, self.y] += 1
        
        # Decay pheromones across the map
        self.pheromone_map = self.pheromone_map * 0.95
        
        # Check if there's a victim in sensed cells
        for cell in sensed_data:
            if cell['type'] == 2:  # Victim
                # Move toward victim
                dx = cell['x'] - self.x
                dy = cell['y'] - self.y
                
                move_x = 0 if dx == 0 else dx // abs(dx)
                move_y = 0 if dy == 0 else dy // abs(dy)
                
                if self.environment.is_valid_position(self.x + move_x, self.y + move_y):
                    self.x += move_x
                    self.y += move_y
                    return
        
        # If no victim, move to cell with lowest pheromone level
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        valid_moves = []
        
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if self.environment.is_valid_position(new_x, new_y):
                pheromone_level = self.pheromone_map[new_x, new_y]
                valid_moves.append((new_x, new_y, pheromone_level))
        
        if valid_moves:
            # Sort by pheromone level (ascending)
            valid_moves.sort(key=lambda x: x[2])
            self.x, self.y = valid_moves[0][0], valid_moves[0][1]
    
    def update(self, agents=None):
        """Update agent state and position"""
        # Try to rescue victim at current position
        if self.environment.rescue_victim(self.x, self.y):
            self.rescued_count += 1
        
        # Sense the environment
        sensed_data = self.sense_environment()
        
        # Move based on the selected coordination strategy
        if self.coordination_type == "basic":
            self.basic_coordination(sensed_data)
        elif self.coordination_type == "stigmergy":
            self.stigmergy_coordination(sensed_data, agents)
    
    def draw(self, canvas):
        """Draw the agent on the canvas"""
        canvas.delete(self.name)
        
        # Calculate pixel position
        center_x = (self.x + 0.5) * self.environment.cell_size
        center_y = (self.y + 0.5) * self.environment.cell_size
        half_size = self.size / 2
        
        # Draw agent body
        canvas.create_oval(
            center_x - half_size, center_y - half_size,
            center_x + half_size, center_y + half_size,
            fill=self.color, tags=self.name
        )
        
        # Draw sensor range indicator (optional)
        range_pixels = self.sensor_range * self.environment.cell_size
        canvas.create_oval(
            center_x - range_pixels, center_y - range_pixels,
            center_x + range_pixels, center_y + range_pixels,
            outline="lightblue", width=1, tags=self.name
        )
        
        # Display agent ID
        canvas.create_text(
            center_x, center_y,
            text=self.name[-1],  # Just the number part of the name
            fill="white", tags=self.name
        )


class ExperimentController:
    """Manages experiment setup, execution, and metrics"""
    
    def __init__(self, window, grid_size=20, cell_size=30):
        self.window = window
        self.window.title("Search and Rescue Simulation")
        
        # Create environment
        self.environment = DisasterEnvironment(grid_size, cell_size)
        
        # Canvas setup
        canvas_width = self.environment.width
        canvas_height = self.environment.height + 60  # Extra space for metrics
        self.canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
        self.canvas.pack()
        
        # Agents
        self.agents = []
        
        # Experiment parameters
        self.running = False
        self.update_interval = 100  # milliseconds
        self.current_coordination = "basic"
        
        # UI controls
        self.create_controls()
    
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
        self.coord_var = tk.StringVar(value="basic")
        
        basic_radio = tk.Radiobutton(control_frame, text="Basic", variable=self.coord_var, 
                                    value="basic", command=self.update_coordination)
        basic_radio.pack(side=tk.LEFT, padx=5, pady=5)
        
        stigmergy_radio = tk.Radiobutton(control_frame, text="Stigmergy", variable=self.coord_var, 
                                        value="stigmergy", command=self.update_coordination)
        stigmergy_radio.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Agent count
        tk.Label(control_frame, text="Agents:").pack(side=tk.LEFT, padx=5, pady=5)
        self.agent_count_var = tk.IntVar(value=3)
        agent_count = tk.Spinbox(control_frame, from_=1, to=10, width=2, textvariable=self.agent_count_var)
        agent_count.pack(side=tk.LEFT, padx=5, pady=5)
    
    def update_coordination(self):
        """Update coordination strategy for all agents"""
        self.current_coordination = self.coord_var.get()
        for agent in self.agents:
            agent.coordination_type = self.current_coordination
    
    def setup_environment(self):
        """Create environment with obstacles and victims"""
        self.environment.create_obstacles(obstacle_density=0.15)
        self.environment.create_disaster_zone(victim_count=10)
    
    def create_agents(self, count=3):
        """Create and place agents in the environment"""
        self.agents = []
        
        for i in range(count):
            agent = Agent(f"Agent{i}", self.environment)
            agent.place_at_random_position()
            agent.coordination_type = self.current_coordination
            self.agents.append(agent)
    
    def start_simulation(self):
        """Start the simulation"""
        if not self.running:
            self.running = True
            self.environment.start_mission()
            self.update_simulation()
            self.start_button.config(text="Pause", command=self.pause_simulation)
    
    def pause_simulation(self):
        """Pause the simulation"""
        if self.running:
            self.running = False
            self.start_button.config(text="Resume", command=self.start_simulation)
    
    def reset_simulation(self):
        """Reset the entire simulation"""
        self.running = False
        self.start_button.config(text="Start", command=self.start_simulation)
        
        # Reset environment
        self.environment = DisasterEnvironment(
            self.environment.grid_size, 
            self.environment.cell_size
        )
        self.setup_environment()
        
        # Create new agents
        self.create_agents(self.agent_count_var.get())
        
        # Redraw
        self.draw()
    
    def update_simulation(self):
        """Update the simulation state"""
        if self.running:
            # Update all agents
            for agent in self.agents:
                agent.update(self.agents)
            
            # Redraw everything
            self.draw()
            
            # Check if mission is complete
            if self.environment.is_mission_complete():
                self.running = False
                self.start_button.config(text="Completed", state=tk.DISABLED)
                elapsed = time.time() - self.environment.start_time
                print(f"Mission completed in {elapsed:.1f} seconds")
                print(f"Coordination strategy: {self.current_coordination}")
                print(f"Agent count: {len(self.agents)}")
                return
            
            # Schedule next update
            self.window.after(self.update_interval, self.update_simulation)
    
    def draw(self):
        """Draw the environment and all agents"""
        self.environment.draw(self.canvas)
        for agent in self.agents:
            agent.draw(self.canvas)


def main():
    # Create the main window
    window = tk.Tk()
    
    # Create controller
    controller = ExperimentController(window)
    
    # Setup environment and agents
    controller.setup_environment()
    controller.create_agents(3)
    
    # Draw initial state
    controller.draw()
    
    # Start the main loop
    window.mainloop()

if __name__ == "__main__":
    main()