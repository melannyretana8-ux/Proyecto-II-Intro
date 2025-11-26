
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



# ============================================================================
# GENERACIÓN DE MAPA
# ============================================================================



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



# ============================================================================
# JUEGO PRINCIPAL
# ============================================================================

