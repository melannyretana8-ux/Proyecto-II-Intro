import tkinter as tk
from game.mapa import Mapa
from game.jugador import Jugador
from game.registro import RegistroJugador

TAM_CASILLA = 40

COLORES = {
    "Camino": "#d0f0c0",
    "Muro": "#444444",
    "Túnel": "#87CEEB",
    "Liana": "#d4a017"}

class RegistroGUI:
    def __init__(self, root, callback):
        self.root = root
        self.callback = callback
        self.root.title("Registro de Jugador")

        tk.Label(root, text="Escribe tu nombre:", font=("Arial", 14)).pack(pady=10)
        self.entrada = tk.Entry(root, font=("Arial", 14))
        self.entrada.pack(pady=10)

        tk.Button(root, text="Iniciar Juego", font=("Arial", 14),
                  command=self.enviar).pack(pady=10)

        self.error = tk.Label(root, text="", fg="red", font=("Arial", 12))
        self.error.pack()

    def enviar(self):
        nombre = self.entrada.get()
        if self.callback(nombre):
            self.root.destroy()
        else:
            self.error.config(text="Por favor ingresa un nombre válido.")


class JuegoGUI:
    def __init__(self, root, nombre_jugador):
        self.root = root
        self.root.title(f"Escapa del Laberinto - Jugador: {nombre_jugador}")
        self.tiempo = 0
        self.nombre_jugador = nombre_jugador
        self.juego_terminado = False

        self.mapa = Mapa()
        self.jugador = Jugador(0, 0)
        self.canvas = tk.Canvas(root, width=400, height=450, bg="white")
        self.canvas.pack()

        self.dibujar_mapa()
        self.dibujar_jugador()
        self.dibujar_hud()
        self.root.after(1000, self.actualizar_tiempo)
        root.bind("<KeyPress>", self.manejar_tecla)

    def dibujar_mapa(self):
        for y in range(10):
            for x in range(10):
                if (x, y) == self.mapa.salida:
                    color = "#9b30ff"
                else:
                    casilla = self.mapa.obtener_casilla(x, y)
                    color = COLORES.get(casilla.nombre, "white")

                self.canvas.create_rectangle(
                    x*TAM_CASILLA, y*TAM_CASILLA + 40,
                    (x+1)*TAM_CASILLA, (y+1)*TAM_CASILLA + 40,
                    fill=color, outline="black")

    def dibujar_jugador(self):
        self.canvas.delete("jugador")

        x = self.jugador.x * TAM_CASILLA
        y = self.jugador.y * TAM_CASILLA + 40

        self.canvas.create_rectangle(
            x, y, x+TAM_CASILLA, y+TAM_CASILLA,
            fill="blue", tags="jugador")

    def dibujar_hud(self):
        self.canvas.delete("hud")

        # Fondo HUD
        self.canvas.create_rectangle(
            0, 0, 400, 40,
            fill="#222", outline="", tags="hud")

        # Nombre del jugador
        self.canvas.create_text(
            10, 20, anchor="w",
            text=f"Jugador: {self.nombre_jugador}",
            fill="white", font=("Arial", 12, "bold"),
            tags="hud")

        # Tiempo
        self.canvas.create_text(
        200, 25,
        anchor="center",
        text=f"Tiempo: {self.tiempo}s",
        fill="white",
        font=("Arial", 11, "bold"),
        tags="hud")

        # Modo
        self.canvas.create_text(
            390, 20, anchor="e",
            text="Modo: Escapa",
            fill="white",
            font=("Arial", 12),
            tags="hud")

        # Barra de energía
        energia_ratio = self.jugador.energia / self.jugador.energia_max
        largo_barra = int(energia_ratio * 100)

        # bordes
        self.canvas.create_rectangle(
            150, 10, 150+102, 30,
            outline="white", width=2, tags="hud")

        # barra interna
        self.canvas.create_rectangle(
            151, 11, 151 + largo_barra, 29,
            fill="#00ff00", outline="", tags="hud")
        
        porcentaje = int(energia_ratio * 100)
        self.canvas.create_text(
            270, 20,
            anchor="center",
            text=f"{porcentaje}%",
            fill="white",
            font=("Arial", 8, "bold"),
            tags="hud")


    def manejar_tecla(self, event):
        if self.juego_terminado:
            return

        tecla = event.keysym.lower()
        dx = dy = 0

        if tecla in ("w", "up"): dy = -1
        elif tecla in ("s", "down"): dy = 1
        elif tecla in ("a", "left"): dx = -1
        elif tecla in ("d", "right"): dx = 1

        nueva_x = self.jugador.x + dx
        nueva_y = self.jugador.y + dy

        casilla_destino = self.mapa.obtener_casilla(nueva_x, nueva_y)
        self.jugador.mover(dx, dy, casilla_destino)
        self.dibujar_jugador()
        self.jugador.regenerar_energia()
        self.dibujar_hud()

        # Revisa la victoria
        if (self.jugador.x, self.jugador.y) == self.mapa.salida:
            self.ganar()

    def ganar(self):
        self.juego_terminado = True
        self.canvas.create_rectangle(0, 0, 400, 450, fill="black")
        self.canvas.create_text(
            200, 200,
            text="¡HAS LOGRADO ESCAPAR!\n🎉😊",
            fill="white",
            font=("Arial", 26, "bold"))

        self.root.unbind("<KeyPress>")

    def actualizar_tiempo(self):
        if self.juego_terminado:
            return

        self.tiempo += 1
        self.dibujar_hud()
        self.root.after(1000, self.actualizar_tiempo)


def main():
    registro = RegistroJugador()

    def iniciar(nombre):
        return registro.set_nombre(nombre)

    root = tk.Tk()
    RegistroGUI(root, iniciar)
    root.mainloop()

    root2 = tk.Tk()
    JuegoGUI(root2, registro.get_nombre())
    root2.mainloop()


if __name__ == "__main__":
    main()

