# smoke2d_taichi_min.py
import taichi as ti
ti.init(arch=ti.gpu)  # Mac → Metal, Windows → CUDA/Vulkan, і т.д.

N = 128                 # розмір ґратки (N×N)
dt = 0.1                # крок часу
dissipation = 0.999     # згасання щільності (1.0 = не зникає)
jacobi_iters = 40       # ітерацій Якобі для тиску
h = 1.0 / N             # крок сітки

# поля
vel      = ti.Vector.field(2, ti.f32, shape=(N, N))
vel_tmp  = ti.Vector.field(2, ti.f32, shape=(N, N))
dens     = ti.field(ti.f32, shape=(N, N))
dens_tmp = ti.field(ti.f32, shape=(N, N))
div      = ti.field(ti.f32, shape=(N, N))
press    = ti.field(ti.f32, shape=(N, N))
press_n  = ti.field(ti.f32, shape=(N, N))

# --------- утиліти семплінгу (бі-лінійна інтерполяція) ----------
@ti.func
def clamp01(x):  # корисно для нормалізованих коорд
    return ti.max(0.0, ti.min(1.0, x))

@ti.func
def sample_scalar(q, p):  # q: ti.field, p: індексні координати (0..N-1)
    # clamp у [0, N-1.001], щоб i1/j1 не вийшли за межі
    x = ti.min(N - 1.001, ti.max(0.0, p.x))
    y = ti.min(N - 1.001, ti.max(0.0, p.y))

    i0 = ti.cast(ti.floor(x), ti.i32)
    j0 = ti.cast(ti.floor(y), ti.i32)
    i1 = i0 + 1
    j1 = j0 + 1

    sx = x - i0
    sy = y - j0

    a = q[i0, j0]
    b = q[i1, j0]
    c = q[i0, j1]
    d = q[i1, j1]

    return (1 - sx) * (1 - sy) * a + sx * (1 - sy) * b + (1 - sx) * sy * c + sx * sy * d


@ti.func
def sample_vec(q, p):     # q: ti.Vector.field(2, ...), p: індексні координати (0..N-1)
    # ТЕ САМЕ, але a/b/c/d — вектори vec2; лінійна комбінація поверне vec2
    x = ti.min(N - 1.001, ti.max(0.0, p.x))
    y = ti.min(N - 1.001, ti.max(0.0, p.y))

    i0 = ti.cast(ti.floor(x), ti.i32)
    j0 = ti.cast(ti.floor(y), ti.i32)
    i1 = i0 + 1
    j1 = j0 + 1

    sx = x - i0
    sy = y - j0

    a = q[i0, j0]
    b = q[i1, j0]
    c = q[i0, j1]
    d = q[i1, j1]

    return (1 - sx) * (1 - sy) * a + sx * (1 - sy) * b + (1 - sx) * sy * c + sx * sy * d


# --------- кроки симуляції ----------
@ti.kernel
def add_buoyancy(alpha: ti.f32, beta: ti.f32, ambient: ti.f32):
    # v.y += alpha * (dens - ambient) + beta (постійний "вітер" нагору)
    for i, j in ti.ndrange(N, N):
        vel[i, j].y += (alpha * (dens[i, j] - ambient) + beta) * dt

@ti.kernel
def splat(cx: ti.f32, cy: ti.f32, r: ti.f32, d_add: ti.f32, vx: ti.f32, vy: ti.f32):
    # додати "дим" і імпульс у круговій області
    for i, j in ti.ndrange(N, N):
        x = (i + 0.5) / N
        y = (j + 0.5) / N
        dx = x - cx
        dy = y - cy
        w = ti.exp(-(dx*dx + dy*dy) / (r * r) * 3.0)
        dens[i, j] += d_add * w
        vel[i, j] += ti.Vector([vx, vy]) * w

@ti.kernel
def advect_velocity():
    for i, j in ti.ndrange(N, N):
        p = ti.Vector([i + 0.5, j + 0.5])
        u = vel[i, j]
        back = p - dt * u / h     # назад по траєкторії в індексних координатах
        vel_tmp[i, j] = sample_vec(vel, back)

@ti.kernel
def advect_density():
    for i, j in ti.ndrange(N, N):
        p = ti.Vector([i + 0.5, j + 0.5])
        u = vel[i, j]
        back = p - dt * u / h
        dens_tmp[i, j] = sample_scalar(dens, back) * dissipation

@ti.kernel
def enforce_boundaries():
    # no-slip: занулити швидкість на бордерах
    for i in range(N):
        vel[i, 0] = ti.Vector([0.0, 0.0])
        vel[i, N - 1] = ti.Vector([0.0, 0.0])
        vel[0, i] = ti.Vector([0.0, 0.0])
        vel[N - 1, i] = ti.Vector([0.0, 0.0])

@ti.kernel
def compute_divergence():
    for i, j in ti.ndrange(N, N):
        im = ti.max(i - 1, 0)
        ip = ti.min(i + 1, N - 1)
        jm = ti.max(j - 1, 0)
        jp = ti.min(j + 1, N - 1)
        dudx = (vel[ip, j].x - vel[im, j].x) * 0.5 / h
        dvdy = (vel[i, jp].y - vel[i, jm].y) * 0.5 / h
        div[i, j] = dudx + dvdy

@ti.kernel
def jacobi_pressure():
    for i, j in ti.ndrange(N, N):
        im = ti.max(i - 1, 0)
        ip = ti.min(i + 1, N - 1)
        jm = ti.max(j - 1, 0)
        jp = ti.min(j + 1, N - 1)
        press_n[i, j] = 0.25 * (press[im, j] + press[ip, j] + press[i, jm] + press[i, jp] - div[i, j] * h * h)

@ti.kernel
def subtract_gradient():
    for i, j in ti.ndrange(N, N):
        im = ti.max(i - 1, 0)
        ip = ti.min(i + 1, N - 1)
        jm = ti.max(j - 1, 0)
        jp = ti.min(j + 1, N - 1)
        gradp = ti.Vector([(press[ip, j] - press[im, j]) * 0.5 / h,
                           (press[i, jp] - press[i, jm]) * 0.5 / h])
        vel[i, j] -= gradp

def project():
    compute_divergence()
    press.fill(0.0)
    for _ in range(jacobi_iters):
        jacobi_pressure()
        press.copy_from(press_n)  # ping-pong
    subtract_gradient()

def step():
    advect_velocity()
    vel.copy_from(vel_tmp)

    add_buoyancy(alpha=5.0, beta=0.0, ambient=0.0)
    enforce_boundaries()
    project()

    advect_density()
    dens.copy_from(dens_tmp)

# --------- візуалізація (2D текстура) ----------
window = ti.ui.Window("2D Smoke (Stable Fluids)", (N, N))
canvas = window.get_canvas()

# warm-up (JIT + створення пайплайнів, щоб потім не фризило)
for _ in range(2):
    step()
ti.sync()

while window.running:
    # інжект диму мишею
    if window.is_pressed(ti.ui.LMB):
        x, y = window.get_cursor_pos()  # 0..1
        splat(x, y, 0.06, d_add=4.0, vx=0.0, vy=0.0)
    # легкий горизонтальний “вітер” праворуч
    splat(0.05, 0.5, 0.02, d_add=0.0, vx=1.0, vy=0.0)

    step()
    # відобразити щільність як картинку (авто-нормалізація 0..1)
    canvas.set_image(dens)
    window.show()
