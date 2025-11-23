class Terreno:
    """Esta es una clase base para todos los tipos de casillas"""
    def __init__(self, simbolo, nombre):
        self.simbolo = simbolo
        self.nombre = nombre

    def permite_jugador(self):
        """Define si el jugador puede pasar o no"""
        return True

    def permite_enemigo(self):
        """Define si el enemigo puede pasar o no"""
        return True

class Camino(Terreno):
    def __init__(self):
        super().__init__(simbolo=0, nombre="Camino")

class Muro(Terreno):
    def __init__(self):
        super().__init__(simbolo=1, nombre="Muro")

    def permite_jugador(self):
        return False

    def permite_enemigo(self):
        return False

class Tunel(Terreno):
    def __init__(self):
        super().__init__(simbolo=2, nombre="Túnel")

    def permite_enemigo(self):
        return False

class Liana(Terreno):
    def __init__(self):
        super().__init__(simbolo=3, nombre="Liana")

    def permite_jugador(self):
        return False
