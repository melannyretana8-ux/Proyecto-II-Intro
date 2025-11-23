from game.casillas import Camino, Muro, Tunel, Liana

class Mapa:
    def __init__(self):
        # Mapa base es de 10x10
        # 0 = Camino
        # 1 = Muro
        # 2 = Túnel
        # 3 = Liana
        self.salida = (9, 9)
        layout = [
            [0,0,1,0,0,0,3,0,0,0],
            [0,1,0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,0,0,0],
            [1,0,0,0,0,2,1,0,0,0],
            [0,0,1,0,0,0,0,1,0,0],
            [0,3,0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,1,0],
            [0,1,0,0,0,1,0,0,0,0],
            [0,0,0,0,3,0,0,1,0,0],
            [0,0,1,0,0,0,0,0,0,2] ]

        self.matriz = []
        for fila in layout:
            fila_objetos = []
            for celda in fila:
                if celda == 0:
                    fila_objetos.append(Camino())
                elif celda == 1:
                    fila_objetos.append(Muro())
                elif celda == 2:
                    fila_objetos.append(Tunel())
                elif celda == 3:
                    fila_objetos.append(Liana())
            self.matriz.append(fila_objetos)

    def obtener_casilla(self, x, y):
        # Cuando trate de salirse del mapa = muro
        if y < 0 or y >= len(self.matriz) or x < 0 or x >= len(self.matriz[0]):
            return Muro()
        return self.matriz[y][x]

    def imprimir_mapa(self):
        """TODO: Método opcional para ver el mapa en consola"""
        for fila in self.matriz:
            fila_simbolos = [str(c.simbolo) for c in fila]
            print(" ".join(fila_simbolos))
