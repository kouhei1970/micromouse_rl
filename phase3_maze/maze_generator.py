import sys
import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import random

# Add workspace root to path
sys.path.append(os.getcwd())
from common.maze_generator import MicromouseMazeGenerator
from common.mjcf_builder import MJCFBuilder

class RandomMazeGenerator(MicromouseMazeGenerator):
    def __init__(self, width=3, height=3):
        super().__init__()
        self.width = width
        self.height = height
        # Resize wall arrays for the specific maze size
        # v_walls: (width + 1) x height
        self.v_walls = np.ones((width + 1, height), dtype=int)
        # h_walls: width x (height + 1)
        self.h_walls = np.ones((width, height + 1), dtype=int)
        
        self.visited = np.zeros((width, height), dtype=bool)

    def generate_maze(self):
        """
        Generates a random maze using DFS (Depth-First Search).
        """
        # Reset walls to all closed (1)
        self.v_walls.fill(1)
        self.h_walls.fill(1)
        self.visited.fill(False)
        
        # Start DFS from (0,0)
        self._dfs(0, 0)
        
        print(f"Random {self.width}x{self.height} maze generated.")

    def _dfs(self, x, y):
        self.visited[x, y] = True
        
        # Directions: North, East, South, West
        # (dx, dy, wall_type, wall_x, wall_y)
        directions = [
            (0, 1, 'h', x, y+1), # North
            (1, 0, 'v', x+1, y), # East
            (0, -1, 'h', x, y),  # South
            (-1, 0, 'v', x, y)   # West
        ]
        
        random.shuffle(directions)
        
        for dx, dy, w_type, wx, wy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < self.width and 0 <= ny < self.height and not self.visited[nx, ny]:
                # Remove wall between (x,y) and (nx,ny)
                if w_type == 'h':
                    self.h_walls[wx, wy] = 0
                else:
                    self.v_walls[wx, wy] = 0
                
                self._dfs(nx, ny)

    def generate_mjcf(self, filename="micromouse_random.xml"):
        builder = MJCFBuilder(self)
        # Phase 3: Random maze with walls and posts.
        # Start at (0.09, 0.09) which is center of cell (0,0)
        builder.build(filename, include_walls=True, include_posts=True, mouse_pos="0.09 0.09 0.002", mouse_euler="0 0 90")

if __name__ == "__main__":
    generator = RandomMazeGenerator(3, 3)
    generator.generate_maze()
    generator.generate_mjcf("assets/micromouse_random_3x3.xml")
