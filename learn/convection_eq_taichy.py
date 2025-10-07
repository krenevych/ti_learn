import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  # або ti.cpu, якщо без GPU

# Параметри задачі
nx = 256
L = 2.0
dx = L / (nx - 1)
c = 1.0
dt = 0.5 * dx / c   # CFL ~ 0.5 (стабільність для upwind: c*dt/dx <= 1)
sigma = c * dt / dx

# Поля Taichi
u  = ti.field(dtype=ti.f32, shape=nx)
un = ti.field(dtype=ti.f32, shape=nx)

@ti.kernel
def init():
    # u = 1 всюди, а на відрізку [0.5, 1.0] -> 2 (аналог твоєї IC)
    for i in range(nx):
        x = i * dx
        val = 1.0
        if 0.5 <= x <= 1.0:
            val = 2.0
        u[i] = val

@ti.kernel
def copy_u_to_un():
    for i in range(nx):
        un[i] = u[i]

@ti.kernel
def step_upwind_periodic():
    # Періодичні ГУ: сусід зліва (i-1+nx) % nx
    for i in range(nx):
        il = (i - 1 + nx) % nx
        u[i] = un[i] - sigma * (un[i] - un[il])

def draw(gui):
    # Малюємо ламану по точках (x, u)
    arr = u.to_numpy()  # -> numpy масив
    # Нормалізуємо у-вісь в діапазон [0,1] для GUI
    ymin, ymax = 0.5, 2.5
    xs = np.linspace(0.0, 1.0, nx)
    ys = (arr - ymin) / (ymax - ymin)
    # З’єднаємо сусідні точки лініями
    for i in range(nx - 1):
        p0 = (xs[i],   ys[i])
        p1 = (xs[i+1], ys[i+1])
        gui.line(begin=p0, end=p1, radius=2)  # без кольору -> за замовчуванням

def main():
    init()
    gui = ti.GUI("1D Advection (Upwind, Taichi)", res=(900, 400))
    steps_per_frame = 2  # скільки кроків робити між кадрами

    while gui.running:
        for _ in range(steps_per_frame):
            copy_u_to_un()
            step_upwind_periodic()
        gui.clear(0x000000)
        draw(gui)
        gui.show()  # відобразити кадр

if __name__ == "__main__":
    main()
