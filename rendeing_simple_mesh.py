import taichi as ti
import numpy as np

# 1) GPU якщо є (на Mac це Metal)
ti.init(arch=ti.gpu)

# ---- параметри сітки ----
NX, NY = 30, 20          # вузлів по X і Y
W, H = 1.2, 0.8          # розмір «тканини» в світових координатах

N = NX * NY
# кількість ребер: (NX-1)*NY (горизонтальні) + (NY-1)*NX (вертикальні)
E = (NX - 1) * NY + (NY - 1) * NX

# 2) поля Taichi
pos = ti.Vector.field(3, ti.f32, shape=N)         # позиції вузлів (3D, z=0)
edges = ti.field(ti.i32, shape=E * 2)             # ПЛОСКИЙ (E*2,) масив індексів (пари)

# 3) побудова сітки (на CPU один раз)
verts = np.empty((N, 3), dtype=np.float32)
def vid(i, j):  # індекс вузла у плоскому масиві
    return i + j * NX

for j in range(NY):
    for i in range(NX):
        x = (i / (NX - 1) - 0.5) * W
        y = (j / (NY - 1) - 0.5) * H
        verts[vid(i, j)] = (x, y, 0.0)

# ребра: горизонтальні + вертикальні, ПЛОСКО (… a,b, a,b, …)
ed = np.empty((E, 2), dtype=np.int32)
k = 0
# горизонтальні
for j in range(NY):
    for i in range(NX - 1):
        ed[k, 0] = vid(i, j)
        ed[k, 1] = vid(i + 1, j)
        k += 1
# вертикальні
for j in range(NY - 1):
    for i in range(NX):
        ed[k, 0] = vid(i, j)
        ed[k, 1] = vid(i, j + 1)
        k += 1

pos.from_numpy(verts)
edges.from_numpy(ed.reshape(-1))   # важливо: зробити плоский (E*2,)

# 4) вікно/сцена/камера і рендер
window = ti.ui.Window("Rect Cloth Grid", res=(900, 700))
canvas = window.get_canvas()
scene  = window.get_scene()
camera = ti.ui.Camera()

while window.running:
    # камера (дивимось трохи зверху)
    camera.position(0.0, 0.2, 2.2)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)

    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(2, 3, 2), color=(1, 1, 1))

    # сам малюнок: лінії ребер + (опційно) вузлики як частинки
    scene.lines(pos, indices=edges, width=1.5, color=(0.95, 0.95, 0.98))
    # scene.particles(pos, radius=0.005, color=(0.2, 0.6, 1.0))  # розкоментуй щоб бачити вузли

    canvas.scene(scene)
    window.show()
