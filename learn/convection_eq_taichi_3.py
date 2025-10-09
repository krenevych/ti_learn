import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)  # спершу CPU; коли все працює — можеш спробувати ti.metal

# --- параметри ---
nx = 256
L  = 2.0
dx = L / (nx - 1)
c  = 1.0
dt = 0.5 * dx / c       # CFL ~ 0.5 -> стійко для upwind
sigma = c * dt / dx

ymin, ymax = 0.5, 2.5

# --- поля ---
u  = ti.field(dtype=ti.f32, shape=nx)
un = ti.field(dtype=ti.f32, shape=nx)
verts = ti.Vector.field(2, dtype=ti.f32, shape=nx)  # нормалізовані вершини [0,1]^2

@ti.kernel
def init():
    for i in range(nx):
        x = i * dx
        u[i] = 2.0 if 0.5 <= x <= 1.0 else 1.0

@ti.kernel
def step_upwind_periodic():
    for i in range(nx):
        un[i] = u[i]
    for i in range(nx):
        il = (i - 1 + nx) % nx
        u[i] = un[i] - sigma * (un[i] - un[il])

@ti.kernel
def update_vertices():
    for i in range(nx):
        x = ti.cast(i, ti.f32) / ti.cast(nx - 1, ti.f32)   # -> [0,1]
        y = (u[i] - ymin) / (ymax - ymin)                  # -> [0,1]
        y = ti.min(1.0, ti.max(0.0, y))                    # кламп на всяк випадок
        verts[i] = ti.Vector([x, y])

def main():
    init()
    window = ti.ui.Window("1D Advection (Upwind, ti.ui)", (1000, 400))
    canvas = window.get_canvas()

    steps_per_frame = 1
    while window.running and not window.is_pressed(ti.ui.ESCAPE):
        # фізика
        for _ in range(steps_per_frame):
            step_upwind_periodic()
        update_vertices()

        # рендер
        canvas.set_background_color((0.0, 0.0, 0.0))
        pos = verts.to_numpy()                   # <<< ключ: передаємо numpy, не ti.field
        canvas.circles(pos, radius=2.0, color=(1.0, 1.0, 1.0))
        window.show()

if __name__ == "__main__":
    main()
