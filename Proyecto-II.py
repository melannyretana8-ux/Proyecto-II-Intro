
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

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
