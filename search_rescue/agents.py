"""
Agents module for Search and Rescue simulation.
Contains different agent implementations with various coordination strategies.
Uses A* path planning for communicating agents.
"""

import random
import math
import numpy as np
import heapq

class Agent:
    """Base search and rescue agent with changeable coordination strategies"""
    
    def __init__(self, name, environment, x=None, y=None):
        self.name = name
        self.environment = environment
        
        # Initial position
        self.x = x
        self.y = y
        
        # Agent properties
        self.size = environment.cell_size * 0.8
        self.sensor_range = 2 
        self.color = "blue"
        
        # Agent state
        self.rescued_count = 0
        self.visited_cells = set()  
        self.previous_position = None  
        
        # For detecting collisions with other agents
        self.planned_move = None
        
        # Performance metrics
        self.path_length = 0  # Total distance traveled
        self.idle_time = 0    # Time spent not moving
        
        # Coordination strategy
        self.strategy = "basic" 
        
        # Place at random position if location not provided
        if self.x is None or self.y is None:
            self.place_at_random_position()
    
    def place_at_random_position(self):
        """Place agent at a random valid position in the environment"""
        while True:
            x = random.randint(0, self.environment.grid_size - 1)
            y = random.randint(0, self.environment.grid_size - 1)
            if self.environment.is_valid_position(x, y):
                self.x = x
                self.y = y
                self.previous_position = (x, y)
                self.visited_cells.add((x, y))
                break
    
    def sense_environment(self):
        """Get information about nearby cells within sensor range"""
        sensed_data = []
        
        for dx in range(-self.sensor_range, self.sensor_range + 1):
            for dy in range(-self.sensor_range, self.sensor_range + 1):
                # Calculate distance from agent
                distance = math.sqrt(dx**2 + dy**2)
                
                # Skip cells outside sensor range or current position
                if distance > self.sensor_range or (dx == 0 and dy == 0):
                    continue
                
                cell_x = self.x + dx
                cell_y = self.y + dy
                cell_type = self.environment.get_cell_type(cell_x, cell_y)
                
                if cell_type != -1: 
                    sensed_data.append({
                        'x': cell_x,
                        'y': cell_y,
                        'type': cell_type,
                        'distance': distance
                    })
        
        return sensed_data
    
    def detect_agent_collisions(self, agents):
        """Check if planned move would collide with another agent"""
        if self.planned_move is None:
            return False
            
        new_x, new_y = self.planned_move
        
        for agent in agents:
            if agent is not self: 
                if agent.x == new_x and agent.y == new_y:
                    return True
                    
                # Check if agents would swap positions
                if (agent.x == new_x and agent.y == new_y and 
                    agent.previous_position == (self.x, self.y)):
                    return True
        
        return False
    
    def plan_move_basic(self, sensed_data):
        """Plan next move based on sensed data (basic rule-based approach)"""
        # First priority: rescue victims
        victim_cells = [cell for cell in sensed_data if cell['type'] == 2]
        
        if victim_cells:
            # Sort by distance (closest first)
            victim_cells.sort(key=lambda c: c['distance'])
            target = victim_cells[0]
            
            # If adjacent to a victim, move to its position
            if target['distance'] <= 1:
                self.planned_move = (target['x'], target['y'])
                return
            
            # Move toward the closest victim
            dx = target['x'] - self.x
            dy = target['y'] - self.y
            
            # Take a single step in the direction of the victim
            move_x = 0 if dx == 0 else (1 if dx > 0 else -1)
            move_y = 0 if dy == 0 else (1 if dy > 0 else -1)
            
            # Try to move along x or y, not diagonally
            if move_x != 0 and move_y != 0:
                # Choose randomly between x and y movement
                if random.random() < 0.5:
                    move_y = 0
                else:
                    move_x = 0
            
            # Check if the move is valid
            new_x = self.x + move_x
            new_y = self.y + move_y
            
            if self.environment.is_valid_position(new_x, new_y):
                self.planned_move = (new_x, new_y)
                return
        
        # Second priority: explore unvisited cells
        unvisited_cells = [cell for cell in sensed_data 
                          if (cell['x'], cell['y']) not in self.visited_cells 
                          and self.environment.is_valid_position(cell['x'], cell['y'])]
        
        if unvisited_cells:
            # Sort by distance (closest first)
            unvisited_cells.sort(key=lambda c: c['distance'])
            target = unvisited_cells[0]
            
            # Move toward the closest unvisited cell
            dx = target['x'] - self.x
            dy = target['y'] - self.y
            
            # Take a single step in that direction (non-diagonal)
            move_x = 0 if dx == 0 else (1 if dx > 0 else -1)
            move_y = 0 if dy == 0 else (1 if dy > 0 else -1)
            
            # Try to move along x or y, not diagonally
            if move_x != 0 and move_y != 0:
                if random.random() < 0.5:
                    move_y = 0
                else:
                    move_x = 0
            
            new_x = self.x + move_x
            new_y = self.y + move_y
            
            if self.environment.is_valid_position(new_x, new_y):
                self.planned_move = (new_x, new_y)
                return
        
        # Third priority: random walk
        # Avoid going back to previous position if possible
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            
            # Skip previous position unless no other option
            if (new_x, new_y) == self.previous_position:
                continue
                
            if self.environment.is_valid_position(new_x, new_y):
                self.planned_move = (new_x, new_y)
                return
        
        # If all else fails, allow returning to previous position
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if self.environment.is_valid_position(new_x, new_y):
                self.planned_move = (new_x, new_y)
                return
        
        # If no move is possible, stay in place
        self.planned_move = (self.x, self.y)
        self.idle_time += 1
    
    def execute_move(self):
        """Execute the planned move"""
        if self.planned_move:
            # Record previous position
            self.previous_position = (self.x, self.y)
            
            # Update position
            new_x, new_y = self.planned_move
            
            # Calculate path length
            self.path_length += abs(new_x - self.x) + abs(new_y - self.y)
            
            # Update position
            self.x, self.y = new_x, new_y
            
            # Mark as visited
            self.visited_cells.add((self.x, self.y))
            
            # Reset planned move
            self.planned_move = None
    
    def update(self, agents=None):
        """Update agent state and position"""
        # Try to rescue victim at current position
        if self.environment.rescue_victim(self.x, self.y):
            self.rescued_count += 1
        
        # Sense the environment
        sensed_data = self.sense_environment()
        
        # Plan next move based on strategy
        if self.strategy == "basic":
            self.plan_move_basic(sensed_data)
        elif self.strategy == "stigmergy":
            self.plan_move_stigmergy(sensed_data, agents)
        
        # Check for collisions with other agents
        if agents and len(agents) > 1:
            if self.detect_agent_collisions(agents):
                # If collision detected, replan
                if self.strategy == "basic":
                    self.plan_move_basic(sensed_data)
                elif self.strategy == "stigmergy":
                    self.plan_move_stigmergy(sensed_data, agents)
        
        # Execute the move
        self.execute_move()
    
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
        
        # Draw sensor range indicator 
        range_pixels = self.sensor_range * self.environment.cell_size
        canvas.create_oval(
            center_x - range_pixels, center_y - range_pixels,
            center_x + range_pixels, center_y + range_pixels,
            outline="lightblue", width=1, tags=self.name
        )
        
        # Display agent ID
        canvas.create_text(
            center_x, center_y,
            text=self.name[-1], 
            fill="white", tags=self.name
        )


class StigmergyAgent(Agent):
    """Agent using stigmergy-based coordination (indirect communication via environment)"""
    
    def __init__(self, name, environment, x=None, y=None):
        super().__init__(name, environment, x, y)
        self.strategy = "stigmergy"
        self.color = "green" 
        
        # Pheromone map for stigmergy coordination
        self.pheromone_map = np.zeros((environment.grid_size, environment.grid_size))
        self.pheromone_decay = 0.95 
        self.pheromone_strength = 1.0 
    
    def plan_move_stigmergy(self, sensed_data, agents):
        """Stigmergy-based coordination using virtual pheromones"""
        # Leave pheromone in current location
        self.pheromone_map[self.x, self.y] += self.pheromone_strength
        
        # Decay pheromones across the map
        self.pheromone_map = self.pheromone_map * self.pheromone_decay
        
        # Sync pheromone maps between agents of same type
        self.sync_pheromone_maps(agents)
        
        # First priority: still rescue victims if sensed
        victim_cells = [cell for cell in sensed_data if cell['type'] == 2]
        
        if victim_cells:
            # Sort by distance (closest first)
            victim_cells.sort(key=lambda c: c['distance'])
            target = victim_cells[0]
            
            # If adjacent to a victim, move to its position
            if target['distance'] <= 1:
                self.planned_move = (target['x'], target['y'])
                return
            
            # Otherwise, move toward the closest victim
            dx = target['x'] - self.x
            dy = target['y'] - self.y
            
            # Take a single step in the direction of the victim
            move_x = 0 if dx == 0 else (1 if dx > 0 else -1)
            move_y = 0 if dy == 0 else (1 if dy > 0 else -1)
            
            # Try to move along x or y, not diagonally
            if move_x != 0 and move_y != 0:
                if random.random() < 0.5:
                    move_y = 0
                else:
                    move_x = 0
            
            # Check if the move is valid
            new_x = self.x + move_x
            new_y = self.y + move_y
            
            if self.environment.is_valid_position(new_x, new_y):
                self.planned_move = (new_x, new_y)
                return
        
        # Second priority: move to cell with lowest pheromone level
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        valid_moves = []
        
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if self.environment.is_valid_position(new_x, new_y):
                # Get pheromone level at this position
                pheromone_level = self.pheromone_map[new_x, new_y]
                valid_moves.append((new_x, new_y, pheromone_level))
        
        if valid_moves:
            # Sort by pheromone level (ascending)
            valid_moves.sort(key=lambda x: x[2])
            self.planned_move = (valid_moves[0][0], valid_moves[0][1])
            return
        
        # If no valid move found, stay in place
        self.planned_move = (self.x, self.y)
        self.idle_time += 1
    
    def sync_pheromone_maps(self, agents):
        """Synchronize pheromone maps between stigmergy agents"""
        if not agents:
            return
            
        # Only sync with other stigmergy agents
        stigmergy_agents = [a for a in agents if isinstance(a, StigmergyAgent) and a is not self]
        
        if not stigmergy_agents:
            return
            
        # Combine pheromone maps (take maximum value at each position)
        for agent in stigmergy_agents:
            self.pheromone_map = np.maximum(self.pheromone_map, agent.pheromone_map)
    
    def update(self, agents=None):
        """Update agent state and position"""
        # Try to rescue victim at current position
        if self.environment.rescue_victim(self.x, self.y):
            self.rescued_count += 1
        
        # Sense the environment
        sensed_data = self.sense_environment()
        
        # Plan next move
        self.plan_move_stigmergy(sensed_data, agents)
        
        # Check for collisions with other agents
        if agents and len(agents) > 1:
            if self.detect_agent_collisions(agents):
                # If collision detected, replan
                self.plan_move_stigmergy(sensed_data, agents)
        
        # Execute the move
        self.execute_move()


class CommunicatingAgent(Agent):
    """Agent using direct communication for coordination"""
    
    def __init__(self, name, environment, x=None, y=None):
        super().__init__(name, environment, x, y)
        self.strategy = "communication"
        self.color = "purple" 
        
        # Communication range 
        self.comm_range = 5
        
        # Shared knowledge
        self.known_victims = set()  
        self.assigned_victim = None  
        self.planned_path = [] 
    
    def communicate(self, agents):
        """Exchange information with other agents within communication range"""
        if not agents:
            return
            
        # Only communicate with other communicating agents
        comm_agents = [a for a in agents if isinstance(a, CommunicatingAgent) and a is not self]
        
        if not comm_agents:
            return
        
        # Share information with agents in communication range
        for agent in comm_agents:
            # Check if in communication range
            distance = math.sqrt((self.x - agent.x)**2 + (self.y - agent.y)**2)
            
            if distance <= self.comm_range:
                # Share known victims
                self.known_victims.update(agent.known_victims)
                agent.known_victims.update(self.known_victims)
                
                # Coordinate assignments
                self.coordinate_assignments(agent)
    
    def coordinate_assignments(self, other_agent):
        """Coordinate victim assignments with another agent"""
        # Simple coordination: if both agents are heading to the same victim,
        # the closer one keeps the assignment
        if (self.assigned_victim and other_agent.assigned_victim and 
            self.assigned_victim == other_agent.assigned_victim):
            
            # Calculate distances to the victim
            vx, vy = self.assigned_victim
            self_distance = abs(self.x - vx) + abs(self.y - vy)
            other_distance = abs(other_agent.x - vx) + abs(other_agent.y - vy)
            
            # The agent farther away should find a new assignment
            if self_distance > other_distance:
                self.assigned_victim = None
                self.planned_path = []
            else:
                other_agent.assigned_victim = None
                other_agent.planned_path = []
    
    def update_known_victims(self, sensed_data):
        """Update known victims based on sensed data"""
        for cell in sensed_data:
            if cell['type'] == 2:
                self.known_victims.add((cell['x'], cell['y']))
    
    def plan_move_communication(self, sensed_data, agents):
        """Direct communication-based coordination"""
        # Update knowledge based on sensed data
        self.update_known_victims(sensed_data)
        
        # Communicate with other agents
        self.communicate(agents)
        
        # If no assigned victim, find one
        if not self.assigned_victim:
            unassigned_victims = [v for v in self.known_victims 
                                 if not any(a.assigned_victim == v for a in agents if a is not self)]
            
            if unassigned_victims:
                # Find closest unassigned victim
                distances = [(v, abs(self.x - v[0]) + abs(self.y - v[1])) for v in unassigned_victims]
                distances.sort(key=lambda x: x[1])
                
                self.assigned_victim = distances[0][0]
                self.planned_path = self.plan_path_to_victim(self.assigned_victim)
        
        # If we have a planned path, follow it
        if self.planned_path:
            next_pos = self.planned_path[0]
            
            # Check if the move is valid
            if self.environment.is_valid_position(next_pos[0], next_pos[1]):
                self.planned_move = next_pos
                self.planned_path.pop(0)
                return
            else:
                # Replan if path is blocked
                if self.assigned_victim:
                    self.planned_path = self.plan_path_to_victim(self.assigned_victim)
                else:
                    self.planned_path = []
        
        # If no path or victim, fall back to basic movement
        self.plan_move_basic(sensed_data)
    
    def plan_path_to_victim(self, victim_pos):
        """Plan a path to the assigned victim using A* search"""
        start = (self.x, self.y)
        goal = victim_pos
        
        # Initialize open and closed sets
        open_set = []
        closed_set = set()
        
        heapq.heappush(open_set, (0, start))
        
        # Keep track of where each node came from
        came_from = {}
        
        # Cost from start to current node
        g_score = {start: 0}
        
        # Estimated cost from start to goal through current node
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            # Get node with lowest f_score
            current_f, current = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if current == goal:
                # Reconstruct the path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                
                # Reverse to get path from start to goal
                path.reverse()
                return path
            
            # Add current to closed set
            closed_set.add(current)
            
            # Check all neighboring cells
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if out of bounds or an obstacle
                if not self.environment.is_valid_position(neighbor[0], neighbor[1]):
                    continue
                
                # Skip if in closed set
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score.get(current, float('inf')) + 1
                
                # Check if this path is better
                if tentative_g < g_score.get(neighbor, float('inf')):
                    # Update path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    # Add to open set if not already there
                    if not any(neighbor == pos for _, pos in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # If no path is found, return empty path
        return []
    
    def heuristic(self, a, b):
        """Manhattan distance heuristic for A* search"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def update(self, agents=None):
        """Update agent state and position"""
        # Check if current position is the assigned victim
        if self.assigned_victim and (self.x, self.y) == self.assigned_victim:
            # Try to rescue
            if self.environment.rescue_victim(self.x, self.y):
                self.rescued_count += 1
                self.known_victims.discard(self.assigned_victim)
                self.assigned_victim = None
                self.planned_path = []
        elif self.environment.rescue_victim(self.x, self.y):
            # Rescued an unassigned victim
            self.rescued_count += 1
            self.known_victims.discard((self.x, self.y))
        
        # Sense the environment
        sensed_data = self.sense_environment()
        
        # Plan next move
        self.plan_move_communication(sensed_data, agents)
        
        # Check for collisions with other agents
        if agents and len(agents) > 1:
            if self.detect_agent_collisions(agents):
                # If collision detected, replan
                self.plan_move_communication(sensed_data, agents)
        
        # Execute the move
        self.execute_move()