import taichi as ti

ti.init()

N = 4
G = 0.1
eps = 1e-3  # або eps = 1e-2, якщо хочеш більш «м’яку» взаємодію при зближенні
dt = 1e-3   # або зменш на 5–10 разів, якщо ще «трусить»


x = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)  # particle positions
v = ti.Vector.field(2, dtype=ti.f32, shape=N)  # particle velocities
m = ti.field(dtype=ti.f32, shape=N)  # particle positions
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy

vec2 = ti.types.vector(n=2, dtype=ti.f32)

# @ti.dataclass
# class Planet:
#     x: vec2
#     v: vec2
#     m: ti.f32

@ti.kernel
def compute_U():
    for i, j in ti.ndrange(N, N):
        if i < j:
            r = x[i] - x[j]
            # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
            # This is to prevent 1/0 error which can cause wrong derivative
            # U[None] += -1 / r.norm(1e-3)  # U += -1 / |r|
            U[None] += -  G * m[i] * m[j] /  r.norm(1e-3)

# @ti.kernel
# def compute_U():
#     U[None] = 0.0                         # обов'язково занулити перед сумою!
#     for i, j in ti.ndrange(N, N):
#         if i < j:
#             r = x[i] - x[j]
#             s = ti.sqrt(r.dot(r) + eps*eps)
#             U[None] += - G * m[i] * m[j] / s

@ti.kernel
def advance():
    for i in x:
        v[i] += dt * -x.grad[i] / m[i]

    for i in x:
        x[i] += dt * v[i]  # dx/dt = v

def substep():
    with ti.ad.Tape(loss=U):
        # Kernel invocations in this scope will later contribute to partial derivatives of
        # U with respect to input variables such as x.
        compute_U()  # The tape will automatically compute dU/dx and save the results in x.grad
    advance()


@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random(), ti.random()]
        # m[i] = 1 + ti.cast(ti.floor(ti.random(dtype=ti.f32) * 10), ti.i32)

    m[0] = 3
    m[1] = 50
    m[2] = 5
    m[3] = 15



init()
print(m)
gui = ti.GUI('Autodiff gravity', (1200, 1200))
while gui.running:
    for i in range(1):
        substep()
    gui.circles(pos=x.to_numpy(), radius=m.to_numpy())
    gui.show()
