import taichi as ti

ti.init(arch=ti.cpu)  # можна ti.gpu, але для демонстрації достатньо CPU

# параметр і лосс — обидва скаляри; нам потрібні їхні градієнти
x    = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_loss():
    loss[None] = x[None] * x[None]  # L = x^2

# приклад запуску
x[None] = 3.0                      # зафіксували x
ti.ad.clear_all_gradients()        # занулити попередні градієнти (звичка)

with ti.ad.Tape(loss=loss):        # Tape сам "посіє" loss.grad = 1 і зробить backward
    compute_loss()                 # прямий прохід, що записує loss[None]

print("x =", x[None])              # 3.0
print("L =", loss[None])           # 9.0
print("dL/dx =", x.grad[None])     # 6.0 (тобто 2*x)
