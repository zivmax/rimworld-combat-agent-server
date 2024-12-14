from state import StateCollector
from typing import List, Dict
class Sutil:
    def __init__(self) -> None:
        self._state = StateCollector.current_state
        self._grid = (self.map_state.height, self.map_state.width)
    
    def get_grid(self) -> tuple:
        return self._grid

    def get_obstacles(self) -> List[List]:
        obstacles: List[List] = []
        for cell_row in self._state.map_state.cells:
            for cell in cell_row:
                if cell.is_tree or cell.is_wall:
                    obstacles.append([cell.loc.x, cell.loc.y])
        return obstacles
    
    def get_allies_enemies(self):
        allies: List = []
        enemies: List = []
        for _, state in self._state.pawn_states:
            if state.is_ally:
                allies.append(state)
            else:
                enemies.append(state)
        return allies, enemies
