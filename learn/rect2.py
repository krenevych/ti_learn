import taichi as ti
import numpy as np

# автопідбір бекенду
if ti.cuda: ti.init(arch=ti.cuda)
elif ti.metal: ti.init(arch=ti.metal)
elif ti.vulkan: ti.init(arch=ti.vulkan)
else: ti.init(arch=ti.cpu)

gui = ti.GUI("Filled rect (triangles)", res=(600, 600))

# кути прямокутника (нормовані координати [0..1])
x0, y0 = 0.2, 0.2
x1, y1 = 0.8, 0.7

# 4 вершини: (LL, LR, UR, UL)
verts = np.array([
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
], dtype=np.float32)

# 2 трикутники індексами у вершини
idx = np.array([
    [0, 1, 2],
    [0, 2, 3],
], dtype=np.int32)

while gui.running:
    gui.clear(0x112F41)
    # gui.triangles(verts, indices=idx, color=0xFFAA00)  # заливка
    gui.triangle(verts[0], verts[1], verts[2], color=0xFFAA00)
    gui.triangle(verts[0], verts[2], verts[3], color=0xFFAA00)

    gui.show()
