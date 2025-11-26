
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
    
    def update(self, keys, map_grid: List[List[Tile]], current_time: int):
        """Actualiza la posición del jugador con movimiento discreto (sin diagonales)"""
        if not self.alive:
            return
        
        # Movimiento discreto: solo una dirección a la vez, sin diagonales
        move_delay = 150  # Milisegundos entre movimientos (más rápido si corre)
        is_running = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        if is_running and self.energy > 0:
            move_delay = 100  # Movimiento más rápido al correr
        
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
        if (0 <= new_row < rows and 0 <= new_col < cols and 
            map_grid[new_row][new_col].can_player_pass()):
            # Movimiento válido: actualizar posición
            self.row = new_row
            self.col = new_col
            self.x = new_col * TILE_SIZE + MAP_OFFSET_X
            self.y = new_row * TILE_SIZE + MAP_OFFSET_Y
            self.last_move_time = current_time
    
    def place_trap(self, current_time: int, map_grid: List[List[Tile]]) -> bool:
        """Coloca una trampa si es posible"""
        # Verificar cooldown
        if current_time - self.last_trap_time < TRAP_COOLDOWN:
            return False
        
        # Verificar número máximo de trampas activas
        active_traps = [t for t in self.traps if t.active]
        if len(active_traps) >= MAX_TRAPS:
            return False
        
        # Verificar que la casilla sea válida (no muro)
        if not map_grid[self.row][self.col].can_player_pass():
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
    """Clase que representa un enemigo"""
    
    def __init__(self, start_row: int, start_col: int, map_grid: List[List[Tile]]):
        self.row = start_row
        self.col = start_col
        self.x = start_col * TILE_SIZE + MAP_OFFSET_X
        self.y = start_row * TILE_SIZE + MAP_OFFSET_Y
        self.path = []
        self.last_path_update = 0
        self.map_grid = map_grid
        self.alive = True
        self.death_time = 0  # Tiempo cuando murió (para respawn)
    
    def update(self, target_pos: Tuple[int, int], current_time: int, 
               mode: GameMode, map_grid: List[List[Tile]]):
        """Actualiza la posición del enemigo usando pathfinding"""
        if not self.alive:
            return
        
        self.map_grid = map_grid
        
        # Actualizar pathfinding periódicamente
        if current_time - self.last_path_update > ENEMY_PATHFIND_UPDATE:
            current_grid_pos = (self.row, self.col)
            
            if mode == GameMode.ESCAPA:
                # Perseguir al jugador
                self.path = Pathfinder.astar(map_grid, current_grid_pos, target_pos, for_enemy=True)
            else:  # CAZADOR
                # Huir del jugador (buscar posición aleatoria lejos)
                rows, cols = len(map_grid), len(map_grid[0])
                # Intentar moverse hacia la salida o alejarse
                flee_target = self._find_flee_target(target_pos, rows, cols)
                self.path = Pathfinder.astar(map_grid, current_grid_pos, flee_target, for_enemy=True)
            
            self.last_path_update = current_time
        
        # Seguir el camino
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
            
            # Actualizar posición en grid
            self.row = int((self.y - MAP_OFFSET_Y) / TILE_SIZE)
            self.col = int((self.x - MAP_OFFSET_X) / TILE_SIZE)
    
    def _find_flee_target(self, player_pos: Tuple[int, int], rows: int, cols: int) -> Tuple[int, int]:
        """Encuentra un objetivo para huir del jugador"""
        # Buscar posición en el borde opuesto
        candidates = []
        for r in [1, rows - 2]:
            for c in range(1, cols - 1):
                if self.map_grid[r][c].can_enemy_pass():
                    dist = abs(r - player_pos[0]) + abs(c - player_pos[1])
                    candidates.append(((r, c), dist))
        
        for c in [1, cols - 2]:
            for r in range(1, rows - 1):
                if self.map_grid[r][c].can_enemy_pass():
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
        pygame.display.set_caption("Juego - Escapa y Cazador")
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
        self.score_manager = ScoreManager()
        self.start_time = 0
        self.game_over = False
        self.game_over_reason = ""
        self.enemies_captured = 0  # Contador de enemigos capturados en modo Cazador
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
                    self.player.place_trap(adjusted_time, self.map_grid)
                elif event.key == pygame.K_e and self.mode == GameMode.CAZADOR:
                    # Intentar atrapar enemigo cercano en modo Cazador
                    self.try_capture_enemy()
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


          
