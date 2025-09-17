import taichi as ti
import numpy as np

# 1) GPU якщо є (на Mac це Metal)
ti.init(arch=ti.gpu)

# ---- параметри сітки ----
WIDTH, HEIGHT = 6, 10           # вузлів по X і Y
W, H = 1., 1.          # розмір «тканини» в світових координатах

num_triangles = (HEIGHT - 1) * (WIDTH - 1) * 2
# 2) поля Taichi
positions = ti.Vector.field(3, dtype=ti.f32, shape=(HEIGHT, WIDTH))
vertices = ti.Vector.field(3, dtype=ti.f32, shape=HEIGHT * WIDTH)
indices = ti.field(int, shape=num_triangles * 3)
colors = ti.Vector.field(3, dtype=ti.f32, shape=HEIGHT * WIDTH)

@ti.kernel
def update():
    for i, j in ti.ndrange(HEIGHT, WIDTH):
        vertices[i * WIDTH + j] = positions[i, j]

@ti.kernel
def init_position():
    for i, j in positions:
        positions[i, j] = [
            (i / HEIGHT - 0.5) * W,
            (j / WIDTH - 0.5) * H,
            0
        ]

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(HEIGHT - 1, WIDTH - 1):
        quad_id = (i * (WIDTH - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * WIDTH + j
        indices[quad_id * 6 + 1] = (i + 1) * WIDTH + j
        indices[quad_id * 6 + 2] = i * WIDTH + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * WIDTH + j + 1
        indices[quad_id * 6 + 4] = i * WIDTH + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * WIDTH + j

    for i, j in ti.ndrange(HEIGHT, WIDTH):
        # colors[i * NX + j] = (1.0, 0.0, 0.0)
        if (i // 2 + j // 2) % 3 == 0:
            colors[i * WIDTH + j] = (0.22, 0.72, 0.52)
        elif (i // 2 + j // 2) % 3 == 1:
            colors[i * WIDTH + j] = (0, 0.334, 0.52)
        else:
            colors[i * WIDTH + j] = (1, 0.334, 0.52)

# 4) вікно/сцена/камера і рендер
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

init_position()
initialize_mesh_indices()
current_t = 0.0
dt = 1 / 60.0

while window.running:
    current_t += dt

    update()

    # камера (дивимось трохи зверху)
    camera.position(0.0, 0.2, 2.2)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)
    scene.set_camera(camera)

    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(2, 3, 2), color=(1, 1, 1))

    # сам малюнок: лінії ребер + (опційно) вузлики як частинки
    scene.particles(vertices, radius=0.005, color=(0.2, 0.6, 1.0))  # розкоментуй щоб бачити вузли
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    canvas.scene(scene)
    window.show()
