
import pygame
import random
import json
import os
import math
from enum import Enum
from collections import deque
from typing import List, Tuple, Optional

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

# Configuración de ventana
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Configuración del mapa
MAP_ROWS = 20
MAP_COLS = 30
TILE_SIZE = 30
MAP_OFFSET_X = 50
MAP_OFFSET_Y = 100

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
BROWN = (139, 69, 19)
DARK_BROWN = (101, 67, 33)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 150)
RED = (255, 0, 0)
DARK_RED = (150, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)

# Configuración del jugador
PLAYER_SPEED = 2
PLAYER_RUN_SPEED = 4
PLAYER_ENERGY_MAX = 100
PLAYER_ENERGY_RECOVERY = 0.5
PLAYER_ENERGY_COST = 2.0

# Configuración de trampas
MAX_TRAPS = 3
TRAP_COOLDOWN = 5000  # 5 segundos en milisegundos
TRAP_RESPAWN_TIME = 10000  # 10 segundos en milisegundos

# Configuración de enemigos
ENEMY_SPEED = 1.5
ENEMY_PATHFIND_UPDATE = 1000  # Actualizar pathfinding cada 1 segundo
ENEMY_COUNT = 5

# Sistema de puntuación
SCORE_FILE = "scores.json"
POINTS_PER_ENEMY = 100
POINTS_PER_SECOND = 1
POINTS_LOST_PER_ENEMY_ESCAPE = 50

# Configuración modo Cazador
CAZADOR_CAPTURE_DISTANCE = 1.5  # Distancia para atrapar enemigo (en tiles)

# ============================================================================
# ENUMS Y TIPOS
# ============================================================================

class GameMode(Enum):
    ESCAPA = "Escapa"
    CAZADOR = "Cazador"

class GameState(Enum):
    MENU = "menu"
    NAME_INPUT = "name_input"
    MODE_SELECT = "mode_select"
    PLAYING = "playing"
    PAUSED = "paused"
    EXIT_CONFIRM = "exit_confirm"
    GAME_OVER = "game_over"
    HIGH_SCORES = "high_scores"

# ============================================================================
# CLASES DE CASILLAS
# ============================================================================

class TileType(Enum):
    CAMINO = 0  # Accesible para jugador y enemigos
    LIANAS = 1  # Solo enemigos
    TUNEL = 2  # Solo jugador
    MURO = 3   # Bloquea ambos

class Tile:
    """Clase base para casillas del mapa"""
    def __init__(self, tile_type: TileType):
        self.tile_type = tile_type
        self.color = self._get_color()
    
    def _get_color(self):
        """Retorna el color según el tipo de casilla"""
        colors = {
            TileType.CAMINO: DARK_GREEN,
            TileType.LIANAS: BROWN,
            TileType.TUNEL: DARK_BLUE,
            TileType.MURO: DARK_GRAY
        }
        return colors.get(self.tile_type, GRAY)
    
    def can_player_pass(self) -> bool:
        """Verifica si el jugador puede pasar por esta casilla"""
        return self.tile_type in [TileType.CAMINO, TileType.TUNEL]
    
    def can_enemy_pass(self) -> bool:
        """Verifica si un enemigo puede pasar por esta casilla"""
        return self.tile_type in [TileType.CAMINO, TileType.LIANAS]

class Camino(Tile):
    """Camino accesible para jugador y enemigos"""
    def __init__(self):
        super().__init__(TileType.CAMINO)

class Lianas(Tile):
    """Lianas solo accesibles para enemigos"""
    def __init__(self):
        super().__init__(TileType.LIANAS)

class Tunel(Tile):
    """Túnel solo accesible para el jugador"""
    def __init__(self):
        super().__init__(TileType.TUNEL)

class Muro(Tile):
    """Muro que bloquea a ambos"""
    def __init__(self):
        super().__init__(TileType.MURO)

# ============================================================================
# SISTEMA DE PUNTUACIÓN
# ============================================================================

class ScoreManager:
    """Maneja el sistema de puntuación persistente"""
    
    def __init__(self, filename: str = SCORE_FILE):
        self.filename = filename
        self.scores = self._load_scores()
    
    def _load_scores(self) -> dict:
        """Carga las puntuaciones desde el archivo JSON"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error cargando puntuaciones: {e}")
        return {
            GameMode.ESCAPA.value: [],
            GameMode.CAZADOR.value: []
        }
    
    def _save_scores(self):
        """Guarda las puntuaciones en el archivo JSON"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.scores, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando puntuaciones: {e}")
    
    def add_score(self, mode: GameMode, name: str, score: int):
        """Añade una nueva puntuación y mantiene solo el Top 5"""
        mode_key = mode.value
        if mode_key not in self.scores:
            self.scores[mode_key] = []
        
        self.scores[mode_key].append({"name": name, "score": score})
        self.scores[mode_key].sort(key=lambda x: x["score"], reverse=True)
        self.scores[mode_key] = self.scores[mode_key][:5]  # Top 5
        self._save_scores()
    
    def get_top_scores(self, mode: GameMode) -> List[dict]:
        """Retorna el Top 5 de puntuaciones para un modo"""
        mode_key = mode.value
        return self.scores.get(mode_key, [])

# ============================================================================
# PATHFINDING
# ============================================================================

class Pathfinder:
    """Implementa pathfinding para enemigos"""
    
    @staticmethod
    def heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Distancia Manhattan como heurística"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @staticmethod
    def astar(map_grid: List[List[Tile]], start: Tuple[int, int], 
              goal: Tuple[int, int], for_enemy: bool = True, 
              game_mode: GameMode = None) -> Optional[List[Tuple[int, int]]]:
        """Encuentra el camino más corto"""

        rows, cols = len(map_grid), len(map_grid[0])
        
        def can_pass(row: int, col: int) -> bool:
            if row < 0 or row >= rows or col < 0 or col >= cols:
                return False
            tile = map_grid[row][col]
            
            if for_enemy:
                # Enemigos: según el modo del juego
                if game_mode == GameMode.CAZADOR:
                    # En modo Cazador: enemigos pueden usar Camino, Lianas Y Túneles
                    return tile.tile_type in [TileType.CAMINO, TileType.LIANAS, TileType.TUNEL]
                else:
                    # En modo Escapa: enemigos pueden usar Camino y Lianas (NO túneles)
                    return tile.tile_type in [TileType.CAMINO, TileType.LIANAS]
            else:
                # Jugador: según el modo del juego
                if game_mode == GameMode.CAZADOR:
                    # En modo Cazador: jugador puede usar Camino y Lianas (NO túneles)
                    return tile.tile_type in [TileType.CAMINO, TileType.LIANAS]
                else:
                    # En modo Escapa: jugador puede usar Camino y Túneles (NO lianas)
                    return tile.tile_type in [TileType.CAMINO, TileType.TUNEL]
        
        if not can_pass(start[0], start[1]) or not can_pass(goal[0], goal[1]):
            return None
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: Pathfinder.heuristic(start, goal)}
        
        while open_set:
            open_set.sort(key=lambda x: x[0])
            current = open_set.pop(0)[1]
            
            if current == goal:
                # Reconstruir camino
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not can_pass(neighbor[0], neighbor[1]):
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + Pathfinder.heuristic(neighbor, goal)
                    if neighbor not in [x[1] for x in open_set]:
                        open_set.append((f_score[neighbor], neighbor))
        
        return None  # Cuando no se encontró camino
    
    @staticmethod
    def bfs(map_grid: List[List[Tile]], start: Tuple[int, int], 
            goal: Tuple[int, int], for_enemy: bool = True,
            game_mode: GameMode = None) -> Optional[List[Tuple[int, int]]]:
        """BFS alternativo para pathfinding"""
        rows, cols = len(map_grid), len(map_grid[0])
        
        def can_pass(row: int, col: int) -> bool:
            if row < 0 or row >= rows or col < 0 or col >= cols:
                return False
            tile = map_grid[row][col]
            
            if for_enemy:
                # Enemigos: según el modo del juego
                if game_mode == GameMode.CAZADOR:
                    # En modo Cazador: enemigos pueden usar camino, lianas y túneles
                    return tile.tile_type in [TileType.CAMINO, TileType.LIANAS, TileType.TUNEL]
                else:
                    # En modo Escapa: enemigos pueden usar camino y lianas (NO túneles)
                    return tile.tile_type in [TileType.CAMINO, TileType.LIANAS]
            else:
                # Jugador: según el modo del juego
                if game_mode == GameMode.CAZADOR:
                    # En modo Cazador: jugador puede usar camino y lianas (NO túneles)
                    return tile.tile_type in [TileType.CAMINO, TileType.LIANAS]
                else:
                    # En modo Escapa: jugador puede usar camino y túneles (NO lianas)
                    return tile.tile_type in [TileType.CAMINO, TileType.TUNEL]
        
        if not can_pass(start[0], start[1]) or not can_pass(goal[0], goal[1]):
            return None
        
        queue = deque([start])
        visited = {start}
        came_from = {start: None}
        
        while queue:
            current = queue.popleft()
            
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if neighbor in visited or not can_pass(neighbor[0], neighbor[1]):
                    continue
                
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
        
        return None

# ============================================================================
# GENERACIÓN DE MAPA
# ============================================================================

class MapGenerator:
    """Genera mapas aleatorios con garantía de camino válido"""
    
    @staticmethod
    def generate_map(rows: int, cols: int) -> Tuple[List[List[Tile]], Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
        """Retorna: (mapa, posición_inicial, posición_salida, posiciones_túneles)"""

        # Inicializa todo como muro
        map_grid = [[Muro() for _ in range(cols)] for _ in range(rows)]
        
        # Crea un camino principal desde inicio hasta salida
        start = (1, 1)
        end = (rows - 2, cols - 2)
        
        # Asegura que el inicio y fin sean caminos
        map_grid[start[0]][start[1]] = Camino()
        map_grid[end[0]][end[1]] = Camino()
        
        # Crea camino principal usando algoritmo de laberinto
        current = start
        visited = {start}
        path = [start]
        
        while current != end:
            neighbors = []
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 < nr < rows - 1 and 0 < nc < cols - 1:
                    neighbors.append((nr, nc))
            
            unvisited = [n for n in neighbors if n not in visited]
            
            if unvisited:
                next_pos = random.choice(unvisited)
                map_grid[next_pos[0]][next_pos[1]] = Camino()
                visited.add(next_pos)
                path.append(next_pos)
                current = next_pos
            else:
                # Backtrack
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                else:
                    # Fuerza camino directo si no hay más opciones
                    if current[0] < end[0]:
                        current = (current[0] + 1, current[1])
                    elif current[1] < end[1]:
                        current = (current[0], current[1] + 1)
                    if current not in visited:
                        map_grid[current[0]][current[1]] = Camino()
                        visited.add(current)
                        path.append(current)
        
        # Mejora distribución de tipos de casillas
        # Primero, crear una mejor distribución de caminos
        for _ in range(rows * cols // 4):
            r = random.randint(1, rows - 2)
            c = random.randint(1, cols - 2)
            
            if map_grid[r][c].tile_type == TileType.MURO:
                map_grid[r][c] = Camino()
        
        # Distribulle lianas (solo enemigos)
        liana_count = rows * cols // 10
        for _ in range(liana_count):
            r = random.randint(1, rows - 2)
            c = random.randint(1, cols - 2)
            
            if map_grid[r][c].tile_type == TileType.MURO:
                map_grid[r][c] = Lianas()
        
        # Distribulle túneles (solo jugador), al menos 3-5 túneles
        tunnel_count = max(3, rows * cols // 15)
        tunnel_positions = []
        for _ in range(tunnel_count):
            attempts = 0
            while attempts < 50:
                r = random.randint(1, rows - 2)
                c = random.randint(1, cols - 2)
                
                # Asegura que no esté en el camino principal ni en inicio/fin
                if (map_grid[r][c].tile_type == TileType.MURO and 
                    (r, c) != start and (r, c) != end):
                    map_grid[r][c] = Tunel()
                    tunnel_positions.append((r, c))
                    break
                attempts += 1
        
        # Verifica que existe camino válido (usar modo Escapa por defecto para validación)
        path_check = Pathfinder.astar(map_grid, start, end, for_enemy=False, game_mode=GameMode.ESCAPA)
        if path_check is None:
            # Si no hay camino, crea uno directo
            r, c = start
            while (r, c) != end:
                map_grid[r][c] = Camino()
                if r < end[0]:
                    r += 1
                elif c < end[1]:
                    c += 1
                elif r > end[0]:
                    r -= 1
                elif c > end[1]:
                    c -= 1
        
        return map_grid, start, end, tunnel_positions

# ============================================================================
# TRAMPA
# ============================================================================

class Trap:
    """Representa una trampa colocada por el jugador"""
    def __init__(self, row: int, col: int, placed_time: int):
        self.row = row
        self.col = col
        self.placed_time = placed_time
        self.active = True
        self.triggered = False
        self.trigger_time = 0
    
    def trigger(self, current_time: int):
        """Activa la trampa"""
        self.triggered = True
        self.trigger_time = current_time
        self.active = False
    
    def should_respawn(self, current_time: int) -> bool:
        """Verifica si la trampa debe reaparecer"""
        return self.triggered and (current_time - self.trigger_time) >= TRAP_RESPAWN_TIME
    
    def respawn(self):
        """Reaparece la trampa"""
        self.triggered = False
        self.active = True

# ============================================================================
# JUGADOR
# ============================================================================

class Player:
    """Clase que representa al jugador"""
    
    def __init__(self, start_row: int, start_col: int):
        self.row = start_row
        self.col = start_col
        self.x = start_col * TILE_SIZE + MAP_OFFSET_X
        self.y = start_row * TILE_SIZE + MAP_OFFSET_Y
        self.energy = PLAYER_ENERGY_MAX
        self.traps = []
        self.last_trap_time = 0
        self.score = 0
        self.alive = True
        self.move_cooldown = 0  # Cooldown para movimiento discreto
        self.last_move_time = 0
    
    def update(self, keys, map_grid: List[List[Tile]], current_time: int, game_mode: GameMode = None):
        """Actualiza la posición del jugador con movimiento discreto (sin diagonales)"""
        if not self.alive:
            return
        
        # Movimiento discreto: solo una dirección a la vez, sin diagonales
        move_delay = 150  # Milisegundos entre movimientos (más rápido si corre)
        is_running = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        # Verificar si tiene energía para correr
        if is_running and self.energy > 0:
            move_delay = 100  # Movimiento más rápido al correr
        elif is_running and self.energy <= 0:
            # No puede correr sin energía, usar velocidad normal
            is_running = False
        
        # Verificar si puede moverse (cooldown)
        if current_time - self.last_move_time < move_delay:
            return
        
        # Determinar dirección (solo una a la vez, sin diagonales)
        new_row = self.row
        new_col = self.col
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            new_row = self.row - 1
            new_col = self.col
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            new_row = self.row + 1
            new_col = self.col
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            new_row = self.row
            new_col = self.col - 1
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            new_row = self.row
            new_col = self.col + 1
        else:
            # No se presionó ninguna tecla, recuperar energía
            self.energy += PLAYER_ENERGY_RECOVERY
            if self.energy > PLAYER_ENERGY_MAX:
                self.energy = PLAYER_ENERGY_MAX
            return
        
        # Consumir energía si está corriendo
        if is_running and self.energy > 0:
            self.energy -= PLAYER_ENERGY_COST
            if self.energy < 0:
                self.energy = 0
        else:
            # Recuperar energía
            self.energy += PLAYER_ENERGY_RECOVERY
            if self.energy > PLAYER_ENERGY_MAX:
                self.energy = PLAYER_ENERGY_MAX
        
        # Verificar colisión con el mapa
        rows, cols = len(map_grid), len(map_grid[0])
        if 0 <= new_row < rows and 0 <= new_col < cols:
            tile = map_grid[new_row][new_col]
            # En modo Cazador: puede usar lianas pero NO túneles
            # En modo Escapa: puede usar túneles pero NO lianas
            can_pass = False
            if game_mode == GameMode.CAZADOR:
                # Modo Cazador: Camino y Lianas (NO túneles)
                can_pass = tile.tile_type in [TileType.CAMINO, TileType.LIANAS]
            else:
                # Modo Escapa: Camino y Túneles (NO lianas)
                can_pass = tile.tile_type in [TileType.CAMINO, TileType.TUNEL]
            
            if can_pass:
                # Movimiento válido: actualizar posición
                self.row = new_row
                self.col = new_col
                self.x = new_col * TILE_SIZE + MAP_OFFSET_X
                self.y = new_row * TILE_SIZE + MAP_OFFSET_Y
                self.last_move_time = current_time
    
    def place_trap(self, current_time: int, map_grid: List[List[Tile]], game_mode: GameMode = None) -> bool:
        """Coloca una trampa si es posible"""
        # Verificar cooldown
        if current_time - self.last_trap_time < TRAP_COOLDOWN:
            return False
        
        # Verificar número máximo de trampas activas
        active_traps = [t for t in self.traps if t.active]
        if len(active_traps) >= MAX_TRAPS:
            return False
        
        # Verificar que la casilla sea válida según el modo
        tile = map_grid[self.row][self.col]
        can_place = False
        if game_mode == GameMode.CAZADOR:
            # Modo Cazador: puede colocar en Camino y Lianas
            can_place = tile.tile_type in [TileType.CAMINO, TileType.LIANAS]
        else:
            # Modo Escapa: puede colocar en Camino y Túneles
            can_place = tile.tile_type in [TileType.CAMINO, TileType.TUNEL]
        
        if not can_place:
            return False
        
        # Verificar que no haya otra trampa en la misma posición
        for trap in self.traps:
            if trap.row == self.row and trap.col == self.col and trap.active:
                return False
        
        # Colocar trampa
        trap = Trap(self.row, self.col, current_time)
        self.traps.append(trap)
        self.last_trap_time = current_time
        return True
    
    def update_traps(self, current_time: int):
        """Actualiza el estado de las trampas"""
        for trap in self.traps:
            if trap.should_respawn(current_time):
                trap.respawn()
    
    def get_trap_at(self, row: int, col: int) -> Optional[Trap]:
        """Retorna la trampa en una posición específica si existe y está activa"""
        for trap in self.traps:
            if trap.row == row and trap.col == col and trap.active:
                return trap
        return None

# ============================================================================
# ENEMIGO
# ============================================================================

class Enemy:

    def __init__(self, start_row: int, start_col: int, map_grid: List[List[Tile]]):
        self.row = start_row
        self.col = start_col
        self.x = start_col * TILE_SIZE + MAP_OFFSET_X
        self.y = start_row * TILE_SIZE + MAP_OFFSET_Y
        self.path = []
        self.last_path_update = 0
        self.map_grid = map_grid
        self.alive = True
        self.death_time = 0  # Tiempo cuando murió (para el respawn)
    
    def update(self, target_pos: Tuple[int, int], current_time: int, 
               mode: GameMode, map_grid: List[List[Tile]]):
        """Actualiza la posición del enemigo usando pathfinding"""
        if not self.alive:
            return
        
        self.map_grid = map_grid
        
        # Actualiza el pathfinding periódicamente
        if current_time - self.last_path_update > ENEMY_PATHFIND_UPDATE:
            current_grid_pos = (self.row, self.col)
            
            if mode == GameMode.ESCAPA:
                # Persegue al jugador (enemigos pueden usar: camino y lianas, NO túneles)
                # Si el jugador está en un túnel, buscar la posición accesible más cercana
                accessible_target = self._find_accessible_target(target_pos, map_grid, mode)
                self.path = Pathfinder.astar(map_grid, current_grid_pos, accessible_target, 
                                            for_enemy=True, game_mode=mode)
            else:  # CAZADOR
                # Huir del jugador (enemigos pueden usar: camino, lianas y túneles)
                rows, cols = len(map_grid), len(map_grid[0])
                # Intenta moverse hacia la salida o alejarse
                flee_target = self._find_flee_target(target_pos, rows, cols, map_grid, mode)
                self.path = Pathfinder.astar(map_grid, current_grid_pos, flee_target, 
                                            for_enemy=True, game_mode=mode)
            
            self.last_path_update = current_time
        
        # Segue el camino
        if self.path and len(self.path) > 1:
            next_pos = self.path[1]
            target_x = next_pos[1] * TILE_SIZE + MAP_OFFSET_X
            target_y = next_pos[0] * TILE_SIZE + MAP_OFFSET_Y
            
            # Movimiento suave hacia el siguiente nodo
            dx = target_x - self.x
            dy = target_y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 2:
                self.x += (dx / distance) * ENEMY_SPEED
                self.y += (dy / distance) * ENEMY_SPEED
            else:
                self.x = target_x
                self.y = target_y
                self.path.pop(0)
            
            # Actualiza la posición en grid
            self.row = int((self.y - MAP_OFFSET_Y) / TILE_SIZE)
            self.col = int((self.x - MAP_OFFSET_X) / TILE_SIZE)
        elif not self.path or len(self.path) <= 1:
            # Si no hay camino válido (por ejemplo, jugador en túnel), 
            # los enemigos siguen moviéndose aleatoriamente
            if mode == GameMode.ESCAPA:
                # Movimiento aleatorio cuando no hay path válido
                if current_time - self.last_path_update > ENEMY_PATHFIND_UPDATE * 2:
                    # Intentar moverse en una dirección aleatoria válida
                    rows, cols = len(map_grid), len(map_grid[0])
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    random.shuffle(directions)
                    
                    for dr, dc in directions:
                        new_row = self.row + dr
                        new_col = self.col + dc
                        
                        if 0 <= new_row < rows and 0 <= new_col < cols:
                            tile = map_grid[new_row][new_col]
                            # En modo Escapa: solo Camino y Lianas (NO túneles)
                            if tile.tile_type in [TileType.CAMINO, TileType.LIANAS]:
                                self.row = new_row
                                self.col = new_col
                                self.x = new_col * TILE_SIZE + MAP_OFFSET_X
                                self.y = new_row * TILE_SIZE + MAP_OFFSET_Y
                                self.last_path_update = current_time
                                break
    
    def _find_accessible_target(self, target_pos: Tuple[int, int], 
                                map_grid: List[List[Tile]], mode: GameMode) -> Tuple[int, int]:
        """Encuentra la posición accesible más cercana al objetivo si está en un túnel"""
        rows, cols = len(map_grid), len(map_grid[0])
        target_tile = map_grid[target_pos[0]][target_pos[1]]
        
        # Si el objetivo está en un túnel (modo Escapa), buscar posición accesible cercana
        if mode == GameMode.ESCAPA and target_tile.tile_type == TileType.TUNEL:
            # Buscar la casilla accesible más cercana al túnel
            best_pos = None
            min_distance = float('inf')
            
            # Buscar en un radio alrededor del túnel
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    r = target_pos[0] + dr
                    c = target_pos[1] + dc
                    
                    if 0 <= r < rows and 0 <= c < cols:
                        tile = map_grid[r][c]
                        # Enemigos pueden usar camino y lianas en modo Escapa
                        if tile.tile_type in [TileType.CAMINO, TileType.LIANAS]:
                            distance = abs(dr) + abs(dc)
                            if distance < min_distance:
                                min_distance = distance
                                best_pos = (r, c)
            
            if best_pos:
                return best_pos
        
        # Si el objetivo es accesible, usarlo directamente
        return target_pos
    
    def _find_flee_target(self, player_pos: Tuple[int, int], rows: int, cols: int, 
                         map_grid: List[List[Tile]], mode: GameMode) -> Tuple[int, int]:
        """Encuentra un objetivo para huir del jugador"""
        # Buscar posición en el borde opuesto
        candidates = []
        for r in [1, rows - 2]:
            for c in range(1, cols - 1):
                tile = map_grid[r][c]
                # En modo Cazador: enemigos pueden usar túneles también
                if mode == GameMode.CAZADOR:
                    can_pass = tile.tile_type in [TileType.CAMINO, TileType.LIANAS, TileType.TUNEL]
                else:
                    can_pass = tile.tile_type in [TileType.CAMINO, TileType.LIANAS]
                
                if can_pass:
                    dist = abs(r - player_pos[0]) + abs(c - player_pos[1])
                    candidates.append(((r, c), dist))
        
        for c in [1, cols - 2]:
            for r in range(1, rows - 1):
                tile = map_grid[r][c]
                # En modo Cazador: enemigos pueden usar túneles también
                if mode == GameMode.CAZADOR:
                    can_pass = tile.tile_type in [TileType.CAMINO, TileType.LIANAS, TileType.TUNEL]
                else:
                    can_pass = tile.tile_type in [TileType.CAMINO, TileType.LIANAS]
                
                if can_pass:
                    dist = abs(r - player_pos[0]) + abs(c - player_pos[1])
                    candidates.append(((r, c), dist))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return (rows - 2, cols - 2)  # Fallback
    
    def check_collision_with_player(self, player_row: int, player_col: int) -> bool:
        """Verifica si el enemigo colisiona con el jugador"""
        return self.row == player_row and self.col == player_col
    
    def check_trap_collision(self, trap: Trap) -> bool:
        """Verifica si el enemigo colisiona con una trampa"""
        return self.row == trap.row and self.col == trap.col

# ============================================================================
# JUEGO PRINCIPAL
# ============================================================================

class Game:
    """Clase principal del juego"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Juego, Escapa y Cazador")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.state = GameState.MENU
        self.mode = None
        self.player_name = ""
        self.name_input = ""
        self.map_grid = []
        self.player = None
        self.enemies = []
        self.start_pos = None
        self.exit_pos = None
        self.tunnel_positions = []  # Posiciones de todos los túneles
        self.score_manager = ScoreManager()
        self.start_time = 0
        self.game_over = False
        self.game_over_reason = ""
        self.enemies_captured = 0  # contador de enemigos capturados en modo Cazador
        self.enemies_escaped = 0  # Contador de enemigos que escaparon en modo Cazador
        self.paused_time_offset = 0  # Offset de tiempo cuando se pausa
        self.pause_start_time = 0  # Tiempo cuando se inició la pausa
    
    def run(self):
        """Bucle principal del juego"""
        running = True
        
        while running:
            dt = self.clock.tick(FPS)
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                self.handle_event(event, current_time)
            
            self.update(current_time)
            self.draw()
        
        pygame.quit()
    
    def handle_event(self, event: pygame.event.Event, current_time: int):
        """Maneja eventos de entrada"""
        if event.type == pygame.KEYDOWN:
            if self.state == GameState.NAME_INPUT:
                if event.key == pygame.K_RETURN:
                    if self.name_input.strip():
                        self.player_name = self.name_input.strip()
                        self.state = GameState.MODE_SELECT
                        self.name_input = ""
                elif event.key == pygame.K_BACKSPACE:
                    self.name_input = self.name_input[:-1]
                else:
                    if len(self.name_input) < 20:
                        self.name_input += event.unicode
            
            elif self.state == GameState.MODE_SELECT:
                if event.key == pygame.K_1:
                    self.start_game(GameMode.ESCAPA)
                elif event.key == pygame.K_2:
                    self.start_game(GameMode.CAZADOR)
                elif event.key == pygame.K_ESCAPE:
                    self.state = GameState.NAME_INPUT
                    self.name_input = ""
            
            elif self.state == GameState.PLAYING:
                if event.key == pygame.K_SPACE:
                    # Usar tiempo ajustado para las trampas
                    adjusted_time = current_time - self.paused_time_offset
                    self.player.place_trap(adjusted_time, self.map_grid, self.mode)
                elif event.key == pygame.K_e and self.mode == GameMode.CAZADOR:
                    # Intentar atrapar enemigo cercano en modo Cazador
                    self.try_capture_enemy()
                elif event.key == pygame.K_RETURN:
                    # Teletransporte entre túneles (solo en modo Escapa)
                    if self.mode == GameMode.ESCAPA:
                        self.teleport_between_tunnels()
                elif event.key == pygame.K_ESCAPE:
                    # Pausar y mostrar confirmación de salida
                    self.pause_start_time = current_time
                    self.state = GameState.EXIT_CONFIRM
            
            elif self.state == GameState.EXIT_CONFIRM:
                if event.key == pygame.K_y or event.key == pygame.K_RETURN:
                    # Confirmar salida
                    self.state = GameState.MENU
                    self.reset_game()
                elif event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                    # Cancelar y reanudar
                    self.paused_time_offset += current_time - self.pause_start_time
                    self.state = GameState.PLAYING
            
            elif self.state == GameState.PAUSED:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                    # Reanudar juego
                    self.paused_time_offset += current_time - self.pause_start_time
                    self.state = GameState.PLAYING
            
            elif self.state == GameState.GAME_OVER:
                if event.key == pygame.K_RETURN:
                    self.state = GameState.MENU
                    self.reset_game()
                elif event.key == pygame.K_h:
                    self.state = GameState.HIGH_SCORES
            
            elif self.state == GameState.HIGH_SCORES:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                    self.state = GameState.MENU
            
            elif self.state == GameState.MENU:
                if event.key == pygame.K_RETURN:
                    self.state = GameState.NAME_INPUT
                    self.name_input = ""
                elif event.key == pygame.K_h:
                    self.state = GameState.HIGH_SCORES
        
        # Manejar clics del mouse
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Clic izquierdo
                mouse_pos = pygame.mouse.get_pos()
                if self.state == GameState.PLAYING:
                    # Verificar si se hizo clic en el botón de salir
                    exit_button_rect = pygame.Rect(SCREEN_WIDTH - 150, 10, 140, 40)
                    if exit_button_rect.collidepoint(mouse_pos):
                        self.pause_start_time = current_time
                        self.state = GameState.EXIT_CONFIRM

    def start_game(self, mode: GameMode):
        """Inicia una nueva partida"""
        self.mode = mode
        self.map_grid, self.start_pos, self.exit_pos, self.tunnel_positions = MapGenerator.generate_map(MAP_ROWS, MAP_COLS)
        
        # Crea jugador
        self.player = Player(self.start_pos[0], self.start_pos[1])
        
        # Crea enemigos en posiciones aleatorias válidas
        self.enemies = []
        rows, cols = len(self.map_grid), len(self.map_grid[0])
        enemy_positions = set()
        
        for _ in range(ENEMY_COUNT):
            attempts = 0
            while attempts < 100:
                r = random.randint(1, rows - 2)
                c = random.randint(1, cols - 2)
                
                tile = self.map_grid[r][c]
                # En modo Cazador: enemigos pueden usar túneles también
                can_pass = (tile.tile_type in [TileType.CAMINO, TileType.LIANAS, TileType.TUNEL] 
                           if self.mode == GameMode.CAZADOR 
                           else tile.tile_type in [TileType.CAMINO, TileType.LIANAS])
                if ((r, c) not in enemy_positions and 
                    (r, c) != self.start_pos and 
                    (r, c) != self.exit_pos and
                    can_pass):
                    enemy = Enemy(r, c, self.map_grid)
                    self.enemies.append(enemy)
                    enemy_positions.add((r, c))
                    break
                attempts += 1
        
        self.start_time = pygame.time.get_ticks()
        self.paused_time_offset = 0
        self.pause_start_time = 0
        self.game_over = False
        self.enemies_captured = 0
        self.enemies_escaped = 0
        self.state = GameState.PLAYING
    
    def reset_game(self):
        """Resetea el estado del juego"""
        self.player = None
        self.enemies = []
        self.map_grid = []
        self.mode = None
        self.game_over = False
        self.game_over_reason = ""
        self.enemies_captured = 0
        self.enemies_escaped = 0
        self.paused_time_offset = 0
        self.pause_start_time = 0
    
    def update(self, current_time: int):
        """Actualiza la lógica del juego"""
        # No actualizar si está pausado o en confirmación de salida
        if self.state not in [GameState.PLAYING] or self.game_over:
            return
        
        # Ajusta tiempo considerando pausas
        adjusted_time = current_time - self.paused_time_offset
        
        keys = pygame.key.get_pressed()
        
        # Actualiza jugador
        if self.player:
            self.player.update(keys, self.map_grid, adjusted_time, self.mode)
            self.player.update_traps(adjusted_time)
            
            # Actualiza puntuación (puntos por tiempo)
            elapsed = (adjusted_time - self.start_time) / 1000.0
            self.player.score = int(elapsed * POINTS_PER_SECOND)
        
        # Actualiza enemigos
        if self.player:
            player_pos = (self.player.row, self.player.col)
            rows, cols = len(self.map_grid), len(self.map_grid[0])
            
            for enemy in self.enemies:
                # Respawn de enemigos muertos (solo en modo Escapa)
                if not enemy.alive and self.mode == GameMode.ESCAPA:
                    if adjusted_time - enemy.death_time >= TRAP_RESPAWN_TIME:
                        # Respawn enemigo en posición aleatoria válida
                        attempts = 0
                        while attempts < 100:
                            r = random.randint(1, rows - 2)
                            c = random.randint(1, cols - 2)
                            tile = self.map_grid[r][c]
                            # En modo Cazador: enemigos pueden usar túneles también
                            can_pass = (tile.tile_type in [TileType.CAMINO, TileType.LIANAS, TileType.TUNEL] 
                                       if self.mode == GameMode.CAZADOR 
                                       else tile.tile_type in [TileType.CAMINO, TileType.LIANAS])
                            if (can_pass and 
                                (r, c) != (self.player.row, self.player.col) and
                                (r, c) != self.exit_pos):
                                enemy.row = r
                                enemy.col = c
                                enemy.x = c * TILE_SIZE + MAP_OFFSET_X
                                enemy.y = r * TILE_SIZE + MAP_OFFSET_Y
                                enemy.path = []
                                enemy.alive = True
                                enemy.death_time = 0
                                break
                            attempts += 1
                    continue
                
                if not enemy.alive:
                    continue
                
                # Actualizar posición del enemigo primero
                if self.mode == GameMode.ESCAPA:
                    # Los enemigos siempre se mueven, incluso si el jugador está en un túnel
                    enemy.update(player_pos, adjusted_time, self.mode, self.map_grid)
                    
                    # Verifica colisión con jugador (solo si el jugador NO está en un túnel)
                    player_tile = self.map_grid[self.player.row][self.player.col]
                    if player_tile.tile_type != TileType.TUNEL:
                        if enemy.check_collision_with_player(self.player.row, self.player.col):
                            self.game_over = True
                            self.game_over_reason = "¡Un enemigo te alcanzó!"
                            self.end_game()
                else:  # CAZADOR
                    enemy.update(player_pos, adjusted_time, self.mode, self.map_grid)
                
                # Verifica colisión con trampas DESPUÉS de actualizar posición
                # (Las trampas funcionan en ambos modos)
                if enemy.alive:
                    trap = self.player.get_trap_at(enemy.row, enemy.col)
                    if trap and trap.active:
                        # La trampa elimina al enemigo
                        trap.trigger(adjusted_time)
                        enemy.alive = False
                        enemy.death_time = adjusted_time
                        self.player.score += POINTS_PER_ENEMY
                        if self.mode == GameMode.CAZADOR:
                            self.enemies_captured += 1
                            # Cuando un cazador es atrapado, los demás aparecen en zonas distintas
                            self.respawn_other_enemies(enemy, rows, cols)
                
                # Verifica si enemigo alcanza la salida (solo modo Cazador)
                if self.mode == GameMode.CAZADOR and enemy.alive:
                    if enemy.row == self.exit_pos[0] and enemy.col == self.exit_pos[1]:
                        enemy.alive = False
                        self.enemies_escaped += 1
        
        # Verifica si jugador alcanza la salida (solo en modo Escapa)
        if self.mode == GameMode.ESCAPA and self.player:
            if self.player.row == self.exit_pos[0] and self.player.col == self.exit_pos[1]:
                self.game_over = True
                self.game_over_reason = "¡Escapaste exitosamente!"
                self.end_game()
        
        # Verifica fin del juego en modo Cazador
        if self.mode == GameMode.CAZADOR and self.player:
            alive_enemies = [e for e in self.enemies if e.alive]
            total_enemies = len(self.enemies)
            
            # Si todos los enemigos han sido eliminados (atrapados o escapados)
            if len(alive_enemies) == 0:
                self.game_over = True
                if self.enemies_captured == 0:
                    # No atrapaste ninguno: derrota
                    self.game_over_reason = f"¡Derrota! No atrapaste ningún enemigo. {self.enemies_escaped} escaparon."
                elif self.enemies_captured == total_enemies:
                    # Atrapaste todos: victoria perfecta
                    self.game_over_reason = f"¡Victoria perfecta! Atrapaste todos los {total_enemies} enemigos."
                else:
                    # Atrapaste algunos: victoria parcial
                    self.game_over_reason = f"¡Victoria parcial! Atrapaste {self.enemies_captured} de {total_enemies} enemigos. {self.enemies_escaped} escaparon."
                self.end_game()
    
    def teleport_between_tunnels(self):
        """Teletransporta al jugador entre túneles si está en uno"""
        if not self.player or not self.tunnel_positions:
            return
        
        current_pos = (self.player.row, self.player.col)
        
        # Verifica si el jugador está en un túnel
        if current_pos not in self.tunnel_positions:
            return
        
        # Encontrar otro túnel distinto
        other_tunnels = [t for t in self.tunnel_positions if t != current_pos]
        
        if not other_tunnels:
            return  # No hay otros túneles disponibles
        
        # Seleccionar un túnel aleatorio
        target_tunnel = random.choice(other_tunnels)
        
        # Teletransportar al jugador
        self.player.row, self.player.col = target_tunnel
        self.player.x = target_tunnel[1] * TILE_SIZE + MAP_OFFSET_X
        self.player.y = target_tunnel[0] * TILE_SIZE + MAP_OFFSET_Y
        
        # Forzar a todos los enemigos a actualizar su pathfinding y reubicarlos
        # para que persigan al jugador en su nueva posición
        current_time = pygame.time.get_ticks()
        new_player_pos = (self.player.row, self.player.col)
        
        # Reubicar enemigos en posiciones aleatorias válidas (pero no en túneles)
        rows, cols = len(self.map_grid), len(self.map_grid[0])
        for enemy in self.enemies:
            if enemy.alive:
                # Resetear el tiempo de última actualización para forzar recálculo
                enemy.last_path_update = 0
                # Limpiar el path actual
                enemy.path = []
                
                # Reubicar enemigo en posición aleatoria válida (no en túneles)
                attempts = 0
                while attempts < 50:
                    r = random.randint(1, rows - 2)
                    c = random.randint(1, cols - 2)
                    tile = self.map_grid[r][c]
                    
                    # En modo Escapa: enemigos NO pueden estar en túneles
                    if self.mode == GameMode.ESCAPA:
                        can_spawn = tile.tile_type in [TileType.CAMINO, TileType.LIANAS]
                    else:
                        can_spawn = tile.tile_type in [TileType.CAMINO, TileType.LIANAS, TileType.TUNEL]
                    
                    if (can_spawn and 
                        (r, c) != new_player_pos and
                        (r, c) != self.exit_pos):
                        enemy.row = r
                        enemy.col = c
                        enemy.x = c * TILE_SIZE + MAP_OFFSET_X
                        enemy.y = r * TILE_SIZE + MAP_OFFSET_Y
                        break
                    attempts += 1
                
                # Si estamos en modo Escapa, calcular nuevo path inmediatamente
                if self.mode == GameMode.ESCAPA:
                    current_grid_pos = (enemy.row, enemy.col)
                    # Usar función que encuentra objetivo accesible si jugador está en túnel
                    accessible_target = enemy._find_accessible_target(new_player_pos, self.map_grid, self.mode)
                    enemy.path = Pathfinder.astar(self.map_grid, current_grid_pos, accessible_target, 
                                                 for_enemy=True, game_mode=self.mode)
                    if enemy.path:
                        enemy.last_path_update = current_time
    
    def try_capture_enemy(self):
        """Intenta atrapar un enemigo cercano en modo Cazador"""
        if not self.player or self.mode != GameMode.CAZADOR:
            return
        
        player_x = self.player.x + TILE_SIZE // 2
        player_y = self.player.y + TILE_SIZE // 2
        
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            
            enemy_x = enemy.x + TILE_SIZE // 2
            enemy_y = enemy.y + TILE_SIZE // 2
            
            # Calcular distancia
            distance = math.sqrt((player_x - enemy_x)**2 + (player_y - enemy_y)**2)
            distance_tiles = distance / TILE_SIZE
            
            # Si está suficientemente cerca, atrapar
            if distance_tiles <= CAZADOR_CAPTURE_DISTANCE:
                enemy.alive = False
                self.enemies_captured += 1
                self.player.score += POINTS_PER_ENEMY
                # Cuando un cazador es atrapado, los demás aparecen en zonas distintas
                rows, cols = len(self.map_grid), len(self.map_grid[0])
                self.respawn_other_enemies(enemy, rows, cols)
                break
    
    def respawn_other_enemies(self, captured_enemy, rows: int, cols: int):
        """Reubica a los demás enemigos en zonas distintas cuando uno es capturado"""
        if self.mode != GameMode.CAZADOR:
            return
        
        # Obtener posiciones ocupadas (jugador, salida, enemigo capturado)
        occupied_positions = {
            (self.player.row, self.player.col),
            self.exit_pos,
            (captured_enemy.row, captured_enemy.col)
        }
        
        # Obtener posiciones de otros enemigos vivos
        for other_enemy in self.enemies:
            if other_enemy.alive and other_enemy != captured_enemy:
                occupied_positions.add((other_enemy.row, other_enemy.col))
        
        # Reubicar cada enemigo vivo en una zona distinta
        for enemy in self.enemies:
            if enemy.alive and enemy != captured_enemy:
                attempts = 0
                new_pos = None
                
                while attempts < 200:  # Más intentos para encontrar buena posición
                    # Buscar posición aleatoria lejos del jugador y de otras posiciones ocupadas
                    r = random.randint(1, rows - 2)
                    c = random.randint(1, cols - 2)
                    
                    tile = self.map_grid[r][c]
                    # En modo Cazador: enemigos pueden usar túneles también
                    can_pass = (tile.tile_type in [TileType.CAMINO, TileType.LIANAS, TileType.TUNEL] 
                               if self.mode == GameMode.CAZADOR 
                               else tile.tile_type in [TileType.CAMINO, TileType.LIANAS])
                    
                    # Verificar que sea válida y esté lejos de posiciones ocupadas
                    if (can_pass and 
                        (r, c) not in occupied_positions):
                        
                        # Verificar distancia mínima de otras posiciones ocupadas
                        min_distance = 5  # Distancia mínima en tiles
                        too_close = False
                        for occ_row, occ_col in occupied_positions:
                            dist = abs(r - occ_row) + abs(c - occ_col)
                            if dist < min_distance:
                                too_close = True
                                break
                        
                        if not too_close:
                            new_pos = (r, c)
                            break
                    
                    attempts += 1
                
                # Si encontramos posición, reubicar
                if new_pos:
                    enemy.row, enemy.col = new_pos
                    enemy.x = enemy.col * TILE_SIZE + MAP_OFFSET_X
                    enemy.y = enemy.row * TILE_SIZE + MAP_OFFSET_Y
                    enemy.path = []  # Limpiar pathfinding
                    occupied_positions.add(new_pos)
                else:
                    # Si no encontramos posición lejana, usar cualquier válida
                    attempts = 0
                    while attempts < 100:
                        r = random.randint(1, rows - 2)
                        c = random.randint(1, cols - 2)
                        tile = self.map_grid[r][c]
                        # En modo Cazador: enemigos pueden usar túneles también
                        can_pass = (tile.tile_type in [TileType.CAMINO, TileType.LIANAS, TileType.TUNEL] 
                                   if self.mode == GameMode.CAZADOR 
                                   else tile.tile_type in [TileType.CAMINO, TileType.LIANAS])
                        if (can_pass and 
                            (r, c) not in occupied_positions):
                            enemy.row, enemy.col = r, c
                            enemy.x = c * TILE_SIZE + MAP_OFFSET_X
                            enemy.y = r * TILE_SIZE + MAP_OFFSET_Y
                            enemy.path = []
                            occupied_positions.add((r, c))
                            break
                        attempts += 1
    
    def end_game(self):
        """Finaliza el juego y guarda la puntuación"""
        if self.player and self.player.score > 0:
            self.score_manager.add_score(self.mode, self.player_name, self.player.score)
        self.state = GameState.GAME_OVER
    
    def draw(self):
        """Dibuja todo en pantalla"""
        self.screen.fill(BLACK)
        
        if self.state == GameState.MENU:
            self.draw_menu()
        elif self.state == GameState.NAME_INPUT:
            self.draw_name_input()
        elif self.state == GameState.MODE_SELECT:
            self.draw_mode_select()
        elif self.state == GameState.PLAYING:
            self.draw_game()
        elif self.state == GameState.EXIT_CONFIRM:
            # Dibuja el juego de fondo (pausado)
            self.draw_game()
            # Dibuja diálogo de confirmación encima
            self.draw_exit_confirm()
        elif self.state == GameState.PAUSED:
            # Dibuja el juego de fondo (pausado)
            self.draw_game()
            # Dibuja overlay de pausa
            self.draw_pause_overlay()
        elif self.state == GameState.GAME_OVER:
            self.draw_game_over()
        elif self.state == GameState.HIGH_SCORES:
            self.draw_high_scores()
        
        pygame.display.flip()
    
    def draw_pause_overlay(self):
        """Dibuja el overlay de pausa"""
        # Fondo semitransparente
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        #Texto de pausa
        pause_text = self.font.render("PAUSA", True, YELLOW)
        pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(pause_text, pause_rect)
        
        resume_text = self.small_font.render("Presiona ESC o ENTER para continuar", True, WHITE)
        resume_rect = resume_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        self.screen.blit(resume_text, resume_rect)
    
    def draw_menu(self):
        """Dibuja el menú principal"""
        title = self.font.render("JUEGO - ESCAPA Y CAZADOR", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100))
        self.screen.blit(title, title_rect)
        
        start_text = self.small_font.render("Presiona ENTER para comenzar", True, WHITE)
        start_rect = start_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(start_text, start_rect)
        
        scores_text = self.small_font.render("Presiona H para ver puntuaciones", True, WHITE)
        scores_rect = scores_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
        self.screen.blit(scores_text, scores_rect)
    
    def draw_name_input(self):
        """Dibuja la pantalla de entrada de nombre"""
        title = self.font.render("Ingresa tu nombre:", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        self.screen.blit(title, title_rect)
        
        input_text = self.font.render(self.name_input + "_", True, YELLOW)
        input_rect = input_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(input_text, input_rect)
        
        hint = self.small_font.render("Presiona ENTER para continuar", True, GRAY)
        hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        self.screen.blit(hint, hint_rect)
    
    def draw_mode_select(self):
        """Dibuja la pantalla de selección de modo"""
        title = self.font.render(f"Selecciona modo - {self.player_name}", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 150))
        self.screen.blit(title, title_rect)
        
        mode1 = self.font.render("1 - ESCAPA", True, GREEN)
        mode1_rect = mode1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        self.screen.blit(mode1, mode1_rect)
        
        desc1 = self.small_font.render("Escapa de los enemigos y llega a la salida", True, GRAY)
        desc1_rect = desc1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
        self.screen.blit(desc1, desc1_rect)
        
        mode2 = self.font.render("2 - CAZADOR", True, RED)
        mode2_rect = mode2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        self.screen.blit(mode2, mode2_rect)
        
        desc2 = self.small_font.render("Atrapa enemigos antes de que escapen por la salida", True, GRAY)
        desc2_rect = desc2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
        self.screen.blit(desc2, desc2_rect)
        
        instructions2 = self.small_font.render("Usa E para atrapar o ESPACIO para trampas", True, CYAN)
        instructions2_rect = instructions2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 110))
        self.screen.blit(instructions2, instructions2_rect)
        
        objective2 = self.small_font.render("OBJETIVO: Atrapa al menos un enemigo para ganar", True, YELLOW)
        objective2_rect = objective2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 140))
        self.screen.blit(objective2, objective2_rect)
        
        warning2 = self.small_font.render("Si todos escapan sin atrapar ninguno, pierdes", True, RED)
        warning2_rect = warning2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 170))
        self.screen.blit(warning2, warning2_rect)
    
    def draw_game(self):
        """Dibuja el juego en curso"""
        # Dibuja el mapa
        rows, cols = len(self.map_grid), len(self.map_grid[0])
        for r in range(rows):
            for c in range(cols):
                x = c * TILE_SIZE + MAP_OFFSET_X
                y = r * TILE_SIZE + MAP_OFFSET_Y
                tile = self.map_grid[r][c]
                pygame.draw.rect(self.screen, tile.color, 
                               (x, y, TILE_SIZE, TILE_SIZE))
                pygame.draw.rect(self.screen, BLACK, 
                               (x, y, TILE_SIZE, TILE_SIZE), 1)
                
                # Indicador visual para túneles (solo jugador puede pasar)
                if tile.tile_type == TileType.TUNEL:
                    # Dibujar un símbolo visual más claro para túneles
                    center_x = x + TILE_SIZE // 2
                    center_y = y + TILE_SIZE // 2
                    # Círculo exterior
                    pygame.draw.circle(self.screen, CYAN, 
                                     (center_x, center_y), 
                                     TILE_SIZE // 3, 2)
                    # Círculo interior
                    pygame.draw.circle(self.screen, BLUE, 
                                     (center_x, center_y), 
                                     TILE_SIZE // 6)
                    # Indicador de teletransporte si el jugador está aquí
                    if self.player and (r, c) == (self.player.row, self.player.col):
                        teleport_text = self.small_font.render("ENTER", True, YELLOW)
                        teleport_rect = teleport_text.get_rect(center=(center_x, center_y - TILE_SIZE // 2 - 5))
                        self.screen.blit(teleport_text, teleport_rect)
                
                # Indicador visual para lianas (solo enemigos pueden pasar)
                if tile.tile_type == TileType.LIANAS:
                    # Dibujar líneas diagonales para indicar que es solo para enemigos
                    pygame.draw.line(self.screen, DARK_BROWN, 
                                   (x + 2, y + 2), 
                                   (x + TILE_SIZE - 2, y + TILE_SIZE - 2), 2)
                    pygame.draw.line(self.screen, DARK_BROWN, 
                                   (x + TILE_SIZE - 2, y + 2), 
                                   (x + 2, y + TILE_SIZE - 2), 2)
        
        # Dibuja salida
        exit_x = self.exit_pos[1] * TILE_SIZE + MAP_OFFSET_X
        exit_y = self.exit_pos[0] * TILE_SIZE + MAP_OFFSET_Y
        pygame.draw.rect(self.screen, YELLOW, 
                        (exit_x, exit_y, TILE_SIZE, TILE_SIZE))
        exit_text = self.small_font.render("EXIT", True, BLACK)
        exit_text_rect = exit_text.get_rect(center=(exit_x + TILE_SIZE // 2, 
                                                     exit_y + TILE_SIZE // 2))
        self.screen.blit(exit_text, exit_text_rect)
        
        # Dibuja trampas
        if self.player:
            for trap in self.player.traps:
                if trap.active:
                    trap_x = trap.col * TILE_SIZE + MAP_OFFSET_X
                    trap_y = trap.row * TILE_SIZE + MAP_OFFSET_Y
                    pygame.draw.circle(self.screen, RED, 
                                     (trap_x + TILE_SIZE // 2, 
                                      trap_y + TILE_SIZE // 2), 
                                     TILE_SIZE // 3)
        
        # Dibuja enemigos
        for enemy in self.enemies:
            if enemy.alive:
                pygame.draw.circle(self.screen, RED, 
                                 (int(enemy.x + TILE_SIZE // 2), 
                                  int(enemy.y + TILE_SIZE // 2)), 
                                 TILE_SIZE // 2 - 2)
        
        # Dibuja jugador
        if self.player:
            pygame.draw.circle(self.screen, BLUE, 
                             (int(self.player.x + TILE_SIZE // 2), 
                              int(self.player.y + TILE_SIZE // 2)), 
                             TILE_SIZE // 2 - 2)
        
        # Dibujar botón de salir
        self.draw_exit_button()
        
        # Dibujar HUD
        self.draw_hud()
    
    def draw_exit_button(self):
        """Dibuja el botón de salir en la esquina superior derecha"""
        button_rect = pygame.Rect(SCREEN_WIDTH - 150, 10, 140, 40)
        
        # Color del botón (hover effect)
        mouse_pos = pygame.mouse.get_pos()
        if button_rect.collidepoint(mouse_pos):
            button_color = DARK_RED
        else:
            button_color = RED
        
        pygame.draw.rect(self.screen, button_color, button_rect)
        pygame.draw.rect(self.screen, WHITE, button_rect, 2)
        
        # Texto del botón
        button_text = self.small_font.render("SALIR (ESC)", True, WHITE)
        text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, text_rect)
    
    def draw_exit_confirm(self):
        """Dibuja el diálogo de confirmación de salida"""
        # Fondo semitransparente
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Caja del diálogo
        dialog_width = 500
        dialog_height = 250
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        
        pygame.draw.rect(self.screen, DARK_GRAY, dialog_rect)
        pygame.draw.rect(self.screen, WHITE, dialog_rect, 3)
        
        # Título
        title = self.font.render("¿Salir del juego?", True, YELLOW)
        title_rect = title.get_rect(center=(dialog_x + dialog_width // 2, dialog_y + 50))
        self.screen.blit(title, title_rect)
        
        # Mensaje
        message = self.small_font.render("¿Estás seguro de que deseas salir?", True, WHITE)
        message_rect = message.get_rect(center=(dialog_x + dialog_width // 2, dialog_y + 100))
        self.screen.blit(message, message_rect)
        
        # Opciones
        option1 = self.small_font.render("SÍ (Y o ENTER) - Salir al menú", True, GREEN)
        option1_rect = option1.get_rect(center=(dialog_x + dialog_width // 2, dialog_y + 150))
        self.screen.blit(option1, option1_rect)
        
        option2 = self.small_font.render("NO (N o ESC) - Continuar jugando", True, RED)
        option2_rect = option2.get_rect(center=(dialog_x + dialog_width // 2, dialog_y + 190))
        self.screen.blit(option2, option2_rect)
    
    def draw_hud(self):
        """Dibuja el HUD del juego"""
        if not self.player:
            return
        
        y_offset = 10
        
        # Nombre del jugador
        name_text = self.small_font.render(f"Jugador: {self.player_name}", True, WHITE)
        self.screen.blit(name_text, (10, y_offset))
        y_offset += 30
        
        # Modo de juego
        mode_text = self.small_font.render(f"Modo: {self.mode.value}", True, WHITE)
        self.screen.blit(mode_text, (10, y_offset))
        y_offset += 30
        
        # Puntuación
        score_text = self.small_font.render(f"Puntos: {self.player.score}", True, WHITE)
        self.screen.blit(score_text, (10, y_offset))
        y_offset += 30
        
        # Tiempo (ajustado por pausas)
        current_time = pygame.time.get_ticks()
        adjusted_time = current_time - self.paused_time_offset
        elapsed = (adjusted_time - self.start_time) / 1000.0
        time_text = self.small_font.render(f"Tiempo: {elapsed:.1f}s", True, WHITE)
        self.screen.blit(time_text, (10, y_offset))
        y_offset += 30
        
        # Energía
        energy_text = self.small_font.render(f"Energía: {int(self.player.energy)}/{PLAYER_ENERGY_MAX}", True, WHITE)
        self.screen.blit(energy_text, (10, y_offset))
        y_offset += 20
        
        # Barra de energía
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = y_offset
        pygame.draw.rect(self.screen, DARK_GRAY, (bar_x, bar_y, bar_width, bar_height))
        energy_width = int((self.player.energy / PLAYER_ENERGY_MAX) * bar_width)
        energy_color = GREEN if self.player.energy > 30 else RED
        pygame.draw.rect(self.screen, energy_color, (bar_x, bar_y, energy_width, bar_height))
        pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
        y_offset += 35
        
        # Trampas
        active_traps = len([t for t in self.player.traps if t.active])
        traps_text = self.small_font.render(f"Trampas: {active_traps}/{MAX_TRAPS}", True, WHITE)
        self.screen.blit(traps_text, (10, y_offset))
        y_offset += 30
        
        # Cooldown de trampas
        cooldown_remaining = max(0, TRAP_COOLDOWN - (pygame.time.get_ticks() - self.player.last_trap_time))
        if cooldown_remaining > 0:
            cooldown_text = self.small_font.render(f"Cooldown: {cooldown_remaining/1000:.1f}s", True, YELLOW)
            self.screen.blit(cooldown_text, (10, y_offset))
        
        # Información específica del modo
        if self.mode == GameMode.CAZADOR:
            y_offset += 40
            alive_enemies = len([e for e in self.enemies if e.alive])
            total_enemies = len(self.enemies)
            
            # Enemigos atrapados
            progress_text = self.small_font.render(
                f"Atrapados: {self.enemies_captured}/{total_enemies}", 
                True, GREEN)
            self.screen.blit(progress_text, (10, y_offset))
            y_offset += 25
            
            # Enemigos escapados
            escaped_text = self.small_font.render(
                f"Escapados: {self.enemies_escaped}/{total_enemies}", 
                True, RED)
            self.screen.blit(escaped_text, (10, y_offset))
            y_offset += 25
            
            # Enemigos restantes
            remaining_text = self.small_font.render(
                f"Restantes: {alive_enemies}", 
                True, YELLOW if alive_enemies > 0 else GREEN)
            self.screen.blit(remaining_text, (10, y_offset))
            y_offset += 30
            
            # Objetivo
            if self.enemies_captured == 0 and self.enemies_escaped > 0:
                objective_text = self.small_font.render(
                    "¡PELIGRO! Atrapa al menos uno o perderás", 
                    True, RED)
            else:
                objective_text = self.small_font.render(
                    "OBJETIVO: Atrapa enemigos antes de que escapen", 
                    True, YELLOW)
            self.screen.blit(objective_text, (10, y_offset))
            y_offset += 25
            
            # Instrucciones
            capture_hint = self.small_font.render(
                "E: Atrapar | Espacio: Trampa", 
                True, CYAN)
            self.screen.blit(capture_hint, (10, y_offset))
        
        # Controles
        controls_y = SCREEN_HEIGHT - 140
        controls = [
            "Controles:",
            "WASD / Flechas: Mover (discreto)",
            "Shift: Correr más rápido",
            "Espacio: Colocar trampa",
        ]
        
        if self.mode == GameMode.CAZADOR:
            controls.append("E: Atrapar enemigo cercano")
            controls.append("Terreno: Solo Lianas (NO túneles)")
        else:
            controls.append("ENTER: Teletransporte (en túneles)")
            controls.append("Terreno: Solo Túneles (NO lianas)")
        
        controls.append("ESC: Salir/Menú")
        
        for i, control in enumerate(controls):
            control_text = self.small_font.render(control, True, GRAY)
            self.screen.blit(control_text, (SCREEN_WIDTH - 250, controls_y + i * 25))
    
    def draw_game_over(self):
        """Dibuja la pantalla de fin de juego"""
        title = self.font.render("GAME OVER", True, RED)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100))
        self.screen.blit(title, title_rect)
        
        reason = self.font.render(self.game_over_reason, True, WHITE)
        reason_rect = reason.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        self.screen.blit(reason, reason_rect)
        
        if self.player:
            score_text = self.font.render(f"Puntuación: {self.player.score}", True, YELLOW)
            score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(score_text, score_rect)
        
        continue_text = self.small_font.render("Presiona ENTER para continuar", True, WHITE)
        continue_rect = continue_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100))
        self.screen.blit(continue_text, continue_rect)
        
        scores_text = self.small_font.render("Presiona H para ver puntuaciones", True, GRAY)
        scores_rect = scores_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 130))
        self.screen.blit(scores_text, scores_rect)
    
    def draw_high_scores(self):
        """Dibuja la pantalla de puntuaciones altas"""
        title = self.font.render("PUNTUACIONES ALTAS", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 50))
        self.screen.blit(title, title_rect)
        
        y_offset = 150
        
        # Mostrar puntuaciones para ambos modos
        for mode in [GameMode.ESCAPA, GameMode.CAZADOR]:
            mode_title = self.font.render(f"Modo: {mode.value}", True, YELLOW)
            mode_rect = mode_title.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            self.screen.blit(mode_title, mode_rect)
            y_offset += 50
            
            scores = self.score_manager.get_top_scores(mode)
            if scores:
                for i, score_data in enumerate(scores):
                    score_text = self.small_font.render(
                        f"{i+1}. {score_data['name']}: {score_data['score']}", 
                        True, WHITE)
                    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
                    self.screen.blit(score_text, score_rect)
                    y_offset += 35
            else:
                no_scores = self.small_font.render("Sin puntuaciones aún", True, GRAY)
                no_scores_rect = no_scores.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
                self.screen.blit(no_scores, no_scores_rect)
                y_offset += 35
            
            y_offset += 20
        
        back_text = self.small_font.render("Presiona ESC o ENTER para volver", True, GRAY)
        back_rect = back_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        self.screen.blit(back_text, back_rect)

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

def main():
    """Función principal"""
    try:
        game = Game()
        game.run()
    except Exception as e:
        print(f"Error en el juego: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()