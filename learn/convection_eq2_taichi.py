import taichi as ti
import numpy as np

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


# Функція для нормалізації даних для відображення
def normalize_for_display(u_values, y_min=0.5, y_max=2.5):
    """Нормалізує значення u до діапазону [0, 1] для відображення на екрані"""
    return (u_values - y_min) / (y_max - y_min)


# Ініціалізація
init_field()

# Створення GUI вікна Taichi
window_width = 800
window_height = 600
gui = ti.GUI('1D Advection (Upwind) - Taichi', res=(window_width, window_height))

# Головний цикл
frame_count = 0
paused = False

while gui.running:
    # Обробка подій
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == ti.GUI.SPACE:
            paused = not paused
    
    # Виконуємо симуляцію, якщо не на паузі
    if not paused and frame_count < nt:
        advect_step()
        frame_count += 1
    
    # Підготовка даних для відображення
    u_np = u.to_numpy()
    
    # Створюємо масив точок для лінії графіку
    points = []
    for i in range(nx):
        x_coord = i / (nx - 1)  # нормалізуємо x до [0, 1]
        y_coord = normalize_for_display(u_np[i], 0.5, 2.5)
        # Інвертуємо y для правильного відображення (0 внизу, 1 вгорі)
        points.append([x_coord, y_coord])
    
    # Очищуємо вікно
    gui.clear(0x112F41)
    
    # Малюємо осі координат
    # Вісь X
    gui.line([0.05, 0.1], [0.95, 0.1], radius=2, color=0xFFFFFF)
    # Вісь Y
    gui.line([0.05, 0.1], [0.05, 0.9], radius=2, color=0xFFFFFF)
    
    # Малюємо сітку
    for i in range(5):
        y = 0.1 + i * 0.2
        gui.line([0.05, y], [0.95, y], radius=1, color=0x444444)
    
    for i in range(10):
        x = 0.05 + i * 0.1
        gui.line([x, 0.1], [x, 0.9], radius=1, color=0x444444)
    
    # Малюємо графік як з'єднані лінії
    for i in range(len(points) - 1):
        # Масштабуємо та зміщуємо координати для відображення з полями
        x1 = 0.05 + points[i][0] * 0.9
        y1 = 0.1 + points[i][1] * 0.8
        x2 = 0.05 + points[i + 1][0] * 0.9
        y2 = 0.1 + points[i + 1][1] * 0.8
        gui.line([x1, y1], [x2, y2], radius=2, color=0x068587)
    
    # Малюємо точки на графіку
    for i in range(len(points)):
        x_coord = 0.05 + points[i][0] * 0.9
        y_coord = 0.1 + points[i][1] * 0.8
        gui.circle([x_coord, y_coord], radius=3, color=0xFF0000)
    
    # Відображаємо інформацію
    gui.text(f'Frame: {frame_count}/{nt}', pos=(0.05, 0.95), font_size=20, color=0xFFFFFF)
    gui.text(f'CFL: {sigma:.3f}', pos=(0.05, 0.92), font_size=20, color=0xFFFFFF)
    if paused:
        gui.text('PAUSED (Space to resume)', pos=(0.35, 0.5), font_size=24, color=0xFFFF00)
    else:
        gui.text('Press SPACE to pause, ESC to exit', pos=(0.25, 0.02), font_size=16, color=0xAAAAAA)
    
    # Відображаємо мітки на осях
    gui.text('0.5', pos=(0.01, 0.08), font_size=16, color=0xFFFFFF)
    gui.text('1.5', pos=(0.01, 0.48), font_size=16, color=0xFFFFFF)
    gui.text('2.5', pos=(0.01, 0.88), font_size=16, color=0xFFFFFF)
    
    gui.text('0.0', pos=(0.04, 0.05), font_size=16, color=0xFFFFFF)
    gui.text('1.0', pos=(0.48, 0.05), font_size=16, color=0xFFFFFF)
    gui.text('2.0', pos=(0.92, 0.05), font_size=16, color=0xFFFFFF)
    
    # Оновлюємо вікно
    gui.show()

gui.close()
