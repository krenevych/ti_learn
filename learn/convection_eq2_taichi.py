import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

# Ініціалізація Taichi (можна вибрати ti.gpu або ti.cpu)
ti.init(arch=ti.gpu)  # або ti.cpu якщо GPU недоступний

# Параметри
nx = 61
dx = 2 / (nx - 1)
nt = 120
dt = 0.0125
c = 1.0

sigma = c * dt / dx
print("CFL =", sigma)  # бажано <= 1 для стабільності

# Taichi поля (замість NumPy масивів)
u = ti.field(dtype=ti.f32, shape=nx)
un = ti.field(dtype=ti.f32, shape=nx)

# Сітка для візуалізації
x = np.linspace(0, 2, nx)


@ti.kernel
def init_field():
    """Ініціалізація початкової умови"""
    for i in range(nx):
        u[i] = 1.0
        # Початкова ступінчаста функція
        if i >= int(0.5 / dx) and i <= int(1 / dx):
            u[i] = 2.0


@ti.kernel
def copy_field():
    """Копіювання u -> un"""
    for i in range(nx):
        un[i] = u[i]


@ti.kernel
def advect_upwind():
    """Один крок upwind схеми для рівняння переносу"""
    for i in range(1, nx):
        u[i] = un[i] - sigma * (un[i] - un[i - 1])
    # Гранична умова: залишаємо u[0] як є
    # Для періодичної граничної умови розкоментуйте:
    # u[0] = un[-1] - sigma * (un[-1] - un[-2])


def advect_step():
    """Повний крок адвекції з копіюванням"""
    copy_field()
    advect_upwind()


# Ініціалізація
init_field()

# Візуалізація
plt.ion()
fig, ax = plt.subplots()
# Конвертуємо Taichi field в NumPy для першого відображення
u_np = u.to_numpy()
line, = ax.plot(x, u_np, lw=2, label='Taichi')
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0.5, 2.5)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('1D advection (upwind) - Taichi версія')
ax.legend()
ax.grid(True, alpha=0.3)

# Головний цикл
for n in range(nt):
    advect_step()
    
    # Конвертація Taichi field в NumPy для візуалізації
    u_np = u.to_numpy()
    line.set_ydata(u_np)
    fig.canvas.draw_idle()
    plt.pause(0.01)

plt.ioff()
plt.show()
