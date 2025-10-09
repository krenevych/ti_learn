import taichi as ti

ti.init(arch=ti.cpu)  # спершу CPU; потім можна спробувати ti.metal

nx = 256
H = 220  # "висота" текстури (пікселів)
L = 2.0
dx = L / (nx - 1)
c = 1.0
dt = 0.5 * dx / c  # CFL ~ 0.5
sigma = c * dt / dx

ymin, ymax = 0.5, 2.5

u = ti.field(dtype=ti.f32, shape=nx)
un = ti.field(dtype=ti.f32, shape=nx)

# ГОЛОВНЕ: робимо картинку як (nx, H) -> (width, height)
img = ti.field(dtype=ti.f32, shape=(nx, H))


@ti.kernel
def init():
    for i in range(nx):
        x = i * dx
        u[i] = 2.0 if 0.5 <= x <= 1.0 else 1.0


@ti.kernel
def step():
    for i in range(nx):
        un[i] = u[i]
    for i in range(nx):
        il = (i - 1 + nx) % nx
        u[i] = un[i] - sigma * (un[i] - un[il])  # upwind, періодика


@ti.kernel
def rasterize():
    for x, y in img:
        img[x, y] = 0.0
    for x in range(nx):
        yn = (u[x] - ymin) / (ymax - ymin)  # [0,1]
        # БУЛО: y = int((1.0 - yn) * (H - 1))  <-- це й перевертало
        y = ti.min(H - 1, ti.max(0, int(yn * (H - 1))))
        for t in ti.static(range(-1, 2)):
            y2 = y + t
            if 0 <= y2 < H:
                img[x, y2] = 1.0


def main():
    init()
    window = ti.ui.Window("1D Advection (texture, horizontal)", (900, 360))
    canvas = window.get_canvas()

    while window.running and not window.is_pressed(ti.ui.ESCAPE):
        step()
        rasterize()
        canvas.set_image(img)  # тепер ширина = nx, висота = H
        window.show()


if __name__ == "__main__":
    main()
