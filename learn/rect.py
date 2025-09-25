import taichi as ti

ti.init(arch=ti.metal)

gui = ti.GUI("Rectangle", res=(600, 600))

x0, y0 = 0.01, 0.01   # лівий-нижній кут
x1, y1 = 0.8, 0.7   # правий-верхній кут

while gui.running:
    # gui.clear(0x112F41)
    gui.clear(0xffff00)
    # чотири ребра
    gui.line([x0, y0], [x1, y0], color=0xEEEEF0, radius=2)  # низ
    # gui.line([x1, y0], [x1, y1], color=0xEEEEF0, radius=2)  # праве
    # gui.line([x1, y1], [x0, y1], color=0xEEEEF0, radius=2)  # верх
    # gui.line([x0, y1], [x0, y0], color=0xEEEEF0, radius=2)  # ліве
    gui.rect([x0, y0], [x1, y1], color=0xFFAA00, radius=2)
    gui.show()
