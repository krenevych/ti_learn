import taichi as ti

ti.init(arch=ti.gpu)  # metal на mac, cuda/vulkan на windows

# -------------------------------
# параметри
# -------------------------------
n_particles = 6000
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-4

# константи матеріалу (желе)
E, nu = 5e3, 0.2  # модуль Юнга, коефіцієнт Пуассона
mu = E / (2 * (1 + nu))             # shear modulus
lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame's first parameter

# -------------------------------
# поля
# -------------------------------
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)   # позиції
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)   # швидкості
F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # деформаційний тензор
C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # афінна матриця
Jp = ti.field(dtype=ti.f32, shape=n_particles)              # пластичність (не використаємо тут)

grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))


@ti.kernel
def substep():
    # очистка сітки
    for i, j in grid_m:
        grid_v[i, j] = ti.Vector([0.0, 0.0])
        grid_m[i, j] = 0

    # P2G
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]

        F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
        J = F[p].determinant()

        # Neo-Hookean стрес
        stress = mu * (F[p] @ F[p].transpose() - ti.Matrix.identity(ti.f32, 2)) \
                 + lam * ti.log(J) * ti.Matrix.identity(ti.f32, 2)
        stress = (-dt * 4 * inv_dx * inv_dx) * stress

        mass = 1
        affine = stress + mass * C[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * mass

    # grid step
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * 9.8
            # стінки
            if i < 3 and grid_v[i, j].x < 0: grid_v[i, j].x = 0
            if i > n_grid - 3 and grid_v[i, j].x > 0: grid_v[i, j].x = 0
            if j < 3 and grid_v[i, j].y < 0: grid_v[i, j].y = 0
            if j > n_grid - 3 and grid_v[i, j].y > 0: grid_v[i, j].y = 0

    # G2P
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i][0] * w[j][1]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]


@ti.kernel
def init_particles():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.4]
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        C[i] = ti.Matrix.zero(ti.f32, 2, 2)
        Jp[i] = 1


# запуск
init_particles()
gui = ti.GUI("MPM Jelly", res=(512, 512))

while gui.running:
    for s in range(30):  # кілька субкроків
        substep()
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show()
