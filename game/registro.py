import os

class RegistroJugador:
    def __init__(self):
        self.nombre = ""

    def set_nombre(self, nombre):
        nombre = nombre.strip()
        if nombre == "":
            return False
        self.nombre = nombre
        self.guardar_historial(nombre)
        return True

    def guardar_historial(self, nombre):
        with open("jugadores.txt", "a", encoding="utf-8") as archivo:
            archivo.write(nombre + "\n")

    def get_nombre(self):
        return self.nombre
