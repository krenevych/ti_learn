import taichi as ti

ti.init(arch=ti.cpu)

x = ti.field(ti.f32, shape=(3,), needs_grad=True)
y = ti.field(ti.f32, shape=(3,), needs_grad=True)
loss = ti.field(ti.f32, shape=(), needs_grad=True)


@ti.kernel
def forward():
    for i in x:
        loss[None] = (x[i]) ** 2 + (y[i]) ** 2


for i in range(3):
    x[i] = i - 1
    y[i] = i - 1

print(x)
print(y)

ti.ad.clear_all_gradients()

with ti.ad.Tape(loss=loss):
    forward()

print("L =", loss[None])  # очікуємо 1^2 + 0.5*1^2 = 1.5

for i in range(3):
    print("dL/dx =", x.grad[i])
    print("dL/dy =", y.grad[i])
