import taichi as ti
from taichi import math

ti.init()  # Alternatively, ti.init(arch=ti.cpu)


x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
# x[None] = math.pi / 6.0

@ti.kernel
def compute_y():
    # y[None] = ti.sin(x[None])
    y[None] = ti.cos(x[None])


with ti.ad.Tape(y):
    compute_y()

x[None] = math.pi / 6.0

with ti.ad.Tape(y):
    compute_y()

print(f'f /x={x[None]}/ = {y[None]}')
print(f'dy/dx /x={x[None]}/ = {x.grad[None]}')
