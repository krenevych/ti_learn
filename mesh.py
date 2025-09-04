import taichi as ti
import numpy as np
import trimesh

# -----------------------------
# 1) ІНІЦІАЛІЗАЦІЯ БЕКЕНДУ
# -----------------------------
# Обираємо найкращий доступний бекенд:
if ti.cuda: ti.init(arch=ti.cuda)      # Windows + NVIDIA → CUDA (найшвидше)
elif ti.metal: ti.init(arch=ti.metal)  # macOS на Apple Silicon → Metal
elif ti.vulkan: ti.init(arch=ti.vulkan) # AMD/Intel або кросплатформеність → Vulkan
else: ti.init(arch=ti.cpu)             # запасний варіант

# -----------------------------
# 2) ЗАВАНТАЖЕННЯ ТА НОРМАЛІЗАЦІЯ МОДЕЛІ
# -----------------------------
# trimesh сам «з’їсть» OBJ/PLY/STL/GLB; force='mesh' гарантує сітку (не сцену з інстансами)
# mesh = trimesh.load("suzanne.obj", force='mesh')
mesh = trimesh.load("Armadillo.ply", force='mesh')

# Центруємо модель у початок координат, щоб камера «дивилась» чітко на 0,0,0
mesh.apply_translation(-mesh.centroid)

# Масштабуємо так, щоб найбільша сторона влізла ~в куб [-1,1] або [−0.5,0.5].
# Тут робимо нормалізацію до розміру ~1.0 по max extent — так легше налаштовувати камеру/світло.
extent = (mesh.bounds[1] - mesh.bounds[0]).max()
if extent > 0:
    mesh.apply_scale(1.0 / extent)

# Отримуємо сирі дані
verts_np = mesh.vertices.astype(np.float32)  # (N, 3) вершини у світі моделі
faces_np = mesh.faces.astype(np.int32)       # (F, 3) індекси трикутників

# -----------------------------
# 3) TAICHI-ПОЛЯ ДЛЯ ВЕРШИН І ІНДЕКСІВ
# -----------------------------
# У Taichi поля — це буфери на CPU/GPU, якими оперуємо в ядрах.
n_v = len(verts_np)

# verts_rest — початкові (базові) вершини; з них обчислюємо трансформовані (verts_world)
verts_rest  = ti.Vector.field(3, ti.f32, shape=n_v)
verts_world = ti.Vector.field(3, ti.f32, shape=n_v)

# ГОЛОВНЕ: scene.mesh очікує ПЛОСКИЙ 1D масив індексів довжини F*3.
# Якщо подати shape=(F,3) або Vector.field(3), отримаєш помилку "Field with dim 2 accessed…".
faces_flat_np = faces_np.reshape(-1)            # (F*3,)
indices = ti.field(ti.i32, shape=faces_flat_np.size)

# Копіюємо дані з NumPy → Taichi-поля (один раз; потім працюємо на GPU)
verts_rest.from_numpy(verts_np)
indices.from_numpy(faces_flat_np)

# -----------------------------
# 4) ОБЧИСЛЕННЯ ТРАНСФОРМАЦІЇ НА GPU
# -----------------------------
# Просте обертання навколо осі Y (щоб модель «ожила»).
# Обчислення на GPU-ядрі (kernel), аби не ганяти NumPy що кадр.
@ti.kernel
def apply_rotation(angle: ti.f32):
    c, s = ti.cos(angle), ti.sin(angle)
    for i in range(n_v):          # верхній for у Taichi — масово-паралельний
        p = verts_rest[i]         # читаємо базову вершину
        # Обертання у площині XZ (Y лишаємо як є)
        # [ x' ]   [  c  0  s ] [ x ]
        # [ y' ] = [  0  1  0 ] [ y ]
        # [ z' ]   [ -s  0  c ] [ z ]
        verts_world[i] = ti.Vector([c*p.x + s*p.z, p.y, -s*p.x + c*p.z])

# -----------------------------
# 5) РЕНДЕР У СЦЕНІ
# -----------------------------
window = ti.ui.Window("Mesh viewer", res=(900, 700))
canvas = window.get_canvas()
scene  = window.get_scene()   # або window.get_scene() / ti.ui.Scene() залежно від версії
camera = ti.ui.Camera()

angle = 0.0
while window.running:
    angle += 0.01
    apply_rotation(angle)     # обчислюємо verts_world на GPU

    # Налаштування камери: позиція, точка огляду, up-вектор
    camera.position(0.0, 0.6, 2.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)

    # Прив’язуємо камеру до сцени й підсвічуємо модель
    scene.set_camera(camera)
    scene.ambient_light((0.7, 0.7, 0.7))           # рівномірне навколишнє світло
    scene.point_light(pos=(2, 3, 2), color=(1, 1, 1))  # точкове світло для бликів/об’єму

    # КЛЮЧ: передаємо вершини й індекси (плоскі!) у рендерер.
    # two_sided=True вимикає відсікання зворотних граней (корисно для «тонких» моделей).
    scene.mesh(verts_world, indices=indices, color=(0.7, 0.75, 0.8), two_sided=True)

    canvas.scene(scene)
    window.show()
