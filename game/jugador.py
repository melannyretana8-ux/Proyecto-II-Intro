class Jugador:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energia = 100
        self.energia_max = 100
        self.velocidad_normal = 1
        self.velocidad_corriendo = 2
        self.corriendo = False

    def mover(self, dx, dy, casilla_objetivo):
        """Esta función mueve al jugador siempre que la casilla lo permita"""
        if not casilla_objetivo.permite_jugador():
            return  

        velocidad = self.velocidad_corriendo if self.corriendo else self.velocidad_normal

        if self.corriendo:
            self.energia -= 1
            if self.energia <= 0:
                self.energia = 0
                self.corriendo = False

        self.x += dx * velocidad
        self.y += dy * velocidad

    def activar_correr(self):
        if self.energia > 0:
            self.corriendo = True

    def desactivar_correr(self):
        self.corriendo = False

    def regenerar_energia(self):
        if not self.corriendo and self.energia < self.energia_max:
            self.energia += 0.5
