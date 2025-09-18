# gpu_mesh_viewer.py
import math
import numpy as np
import taichi as ti
import trimesh

# -------- 0) Ініціалізація Taichi (GPU, якщо є) ----------
ti.init(
    # arch=ti.metal,              # або cuda/vulkan на інших платформах
    arch=ti.gpu,              # або cuda/vulkan на інших платформах
    # arch=ti.vulkan,              # або cuda/vulkan на інших платформах
    # arch=ti.cuda,              # або cuda/vulkan на інших платформах
    kernel_profiler=True,       # УВІМКНУТИ ПРОФАЙЛЕР
    # async_mode=False            # (необов’язково) спробуй вимкнути async для чистих метрик
)

print("[Taichi] arch =", ti.lang.impl.current_cfg().arch)

# -------- 1) Завантаження та нормалізація моделі ----------
MODEL_PATH = "suzanne.obj"

def load_mesh(path: str):
    try:
        m = trimesh.load(path, force='mesh')  # OBJ/PLY/STL/GLB
    except Exception as e:
        print(f"[WARN] Не вдалося завантажити '{path}': {e}")
        print("[INFO] Створюю примітивний box як заглушку…")
        m = trimesh.primitives.Box(extents=(1, 1, 1)).to_mesh()
    # центр у (0,0,0)
    m.apply_translation(-m.centroid)
    # масштаб до ~1.0 найбільшої сторони
    extent = (m.bounds[1] - m.bounds[0]).max()
    if extent > 0:
        m.apply_scale(1.0 / extent)
    return m

mesh = load_mesh(MODEL_PATH)

verts_np = mesh.vertices.astype(np.float32)       # (N,3) float32 — дружньо до GPU
faces_np = mesh.faces.astype(np.int32).reshape(-1)  # (F*3,) плоский int32

# -------- 2) Поля Taichi (живуть на GPU) ----------
n_v = verts_np.shape[0]
verts_rest  = ti.Vector.field(3, ti.f32, shape=n_v)   # «базові» вершини
verts_world = ti.Vector.field(3, ti.f32, shape=n_v)   # трансформовані вершини (для рендеру)
indices     = ti.field(ti.i32, shape=faces_np.size)   # плоскі індекси F*3

verts_rest.from_numpy(verts_np)
indices.from_numpy(faces_np)

# «уніформ» — матриця моделі 4x4 (оновлюємо з CPU раз на кадр; це дешево)
model = ti.Matrix.field(4, 4, ti.f32, shape=())

# -------- 3) Ядро трансформації вершин (GPU) ----------
@ti.kernel
def apply_model():
    M = model[None]
    for i in range(n_v):  # паралельно на GPU
        p = verts_rest[i]
        q = M @ ti.Vector([p.x, p.y, p.z, 1.0])
        verts_world[i] = q.xyz  # відкидаємо w

# -------- 4) Вікно / сцена / камера ----------
window = ti.ui.Window("GPU Mesh Viewer (Taichi)", res=(1000, 700))
canvas = window.get_canvas()
scene  = window.get_scene()
camera = ti.ui.Camera()

angle = 0.0

# -------- 5) Головний цикл ----------
frame = 0
while window.running:
    angle += 0.01
    c, s = math.cos(angle), math.sin(angle)

    # Матриця моделі: поворот навколо осі Y
    model_np = np.array([
        [ c, 0,  s, 0],
        [ 0, 1,  0, 0],
        [-s, 0,  c, 0],
        [ 0, 0,  0, 1],
    ], dtype=np.float32)
    model.from_numpy(model_np)  # маленька 4x4 копія на GPU
    apply_model()               # масове обчислення вершин — НА GPU

    # Камера й освітлення
    camera.position(0.0, 0.6, 2.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)
    # можна додати керування мишею:
    # camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

    scene.set_camera(camera)
    scene.ambient_light((0.75, 0.75, 0.75))
    scene.point_light(pos=(2, 3, 2), color=(1, 1, 1))

    # Рендер сітки (two_sided=True — не відсікаємо зворотні грані, зручно для дебагу)
    scene.mesh(verts_world, indices=indices, color=(0.72, 0.76, 0.85), two_sided=True)

    canvas.scene(scene)
    window.show()

    # frame += 1
    # if frame % 300 == 0:
    #     ti.sync()  # ВАЖЛИВО: дочекатися завершення всіх ядер
    #     ti.profiler.print_kernel_profiler_info()  # показати зібрану статистику
    #     ti.profiler.clear_kernel_profiler_info()  # (необов'язково) обнулити лічильники
