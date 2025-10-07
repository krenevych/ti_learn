"""
Простий стабільний LBM симулятор
- Фіксовані безпечні параметри
- Без складних розрахунків Re
- Максимальна стабільність
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button, RadioButtons
import time
from typing import List
from dataclasses import dataclass
import math

matplotlib.use("TkAgg")

# Спроба імпорту CuPy для GPU прискорення
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ GPU доступний через CuPy")
except ImportError:
    CUPY_AVAILABLE = False
    print("❌ GPU недоступний - використовуємо CPU")
    cp = None


@dataclass
class MovingBody:
    """Просте рухоме тіло"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 20.0
    color: str = 'red'


class SimpleLBM:
    """Простий стабільний LBM без складних розрахунків"""

    def __init__(self, nx=600, ny=200, u_inlet=0.1, use_gpu=True):
        self.nx = nx
        self.ny = ny
        self.u_inlet = u_inlet
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

        # D2Q9 параметри
        self.q = 9
        self.c = self.xp.array([
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1]
        ], dtype=self.xp.float32)

        self.w = self.xp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
                               dtype=self.xp.float32)
        self.opposite = self.xp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=self.xp.int32)

        # ФІКСОВАНІ безпечні параметри
        self.tau = 1.0  # Безпечне значення
        self.omega = 1.0 / self.tau
        self.Re_display = 100  # Для відображення

        # Рухомі тіла
        self.moving_bodies: List[MovingBody] = []

        # Ініціалізація полів
        self.initialize_fields()

        # Лічильники
        self.time_step = 0
        self.fps = 0
        self.last_time = time.time()
        self.fps_counter = 0

        print(f"📊 Стабільні параметри: τ = {self.tau:.1f}, ω = {self.omega:.3f}")

    def initialize_fields(self):
        """Ініціалізація полів"""
        self.f = self.xp.zeros((self.q, self.ny, self.nx), dtype=self.xp.float32)
        self.rho = self.xp.ones((self.ny, self.nx), dtype=self.xp.float32)
        self.ux = self.xp.ones((self.ny, self.nx), dtype=self.xp.float32) * self.u_inlet
        self.uy = self.xp.zeros((self.ny, self.nx), dtype=self.xp.float32)

        # Статична перешкода (циліндр)
        self.add_cylinder()

        # Рівноважні розподіли
        self.equilibrium()
        self.f[:] = self.feq[:]

    def add_cylinder(self):
        """Додати циліндричну перешкоду"""
        cx, cy = self.nx // 4, self.ny // 2
        r = self.ny // 10

        y, x = self.xp.meshgrid(self.xp.arange(self.ny), self.xp.arange(self.nx), indexing='ij')
        self.obstacle = (x - cx)**2 + (y - cy)**2 <= r**2

    def equilibrium(self):
        """Рівноважні розподіли"""
        u2 = self.ux**2 + self.uy**2

        if not hasattr(self, 'feq'):
            self.feq = self.xp.zeros_like(self.f)

        for i in range(self.q):
            cu = self.ux * self.c[0, i] + self.uy * self.c[1, i]
            self.feq[i] = self.w[i] * self.rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

    def macroscopic(self):
        """Макроскопічні величини"""
        self.rho = self.xp.sum(self.f, axis=0)

        # Безпечне обчислення
        rho_safe = self.xp.maximum(self.rho, 0.1)
        self.ux = self.xp.sum(self.f * self.c[0, :, None, None], axis=0) / rho_safe
        self.uy = self.xp.sum(self.f * self.c[1, :, None, None], axis=0) / rho_safe

        # Граничні умови
        self.ux[self.obstacle] = 0
        self.uy[self.obstacle] = 0

    def collision(self):
        """BGK колізія"""
        self.equilibrium()
        self.f += self.omega * (self.feq - self.f)

    def streaming(self):
        """Streaming"""
        for i in range(self.q):
            self.f[i] = self.xp.roll(self.f[i], int(self.c[0, i]), axis=1)
            self.f[i] = self.xp.roll(self.f[i], int(self.c[1, i]), axis=0)

    def boundary(self):
        """Граничні умови"""
        # Bounce-back на перешкоді
        for i in range(self.q):
            self.f[i, self.obstacle] = self.f[self.opposite[i], self.obstacle]

        # Вхід
        self.ux[:, 0] = self.u_inlet
        self.uy[:, 0] = 0
        self.rho[:, 0] = 1

        u2 = self.u_inlet**2
        for i in range(self.q):
            cu = self.u_inlet * self.c[0, i]
            self.f[i, :, 0] = self.w[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

        # Вихід
        self.f[:, :, -1] = self.f[:, :, -2]

    def step(self):
        """Один крок симуляції"""
        try:
            # Простий FSI для рухомих тіл
            self.update_bodies()

            # LBM крок
            self.collision()
            self.streaming()
            self.boundary()
            self.macroscopic()

            self.time_step += 1

            # FPS
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.last_time > 1.0:
                self.fps = self.fps_counter / (current_time - self.last_time)
                self.fps_counter = 0
                self.last_time = current_time

        except Exception as e:
            print(f"❌ Помилка: {e}")
            self.reset_fields()

    def update_bodies(self):
        """Простий FSI для рухомих тіл"""
        for body in self.moving_bodies:
            ix, iy = int(body.x), int(body.y)

            # Перевірка меж
            if ix < 5 or ix >= self.nx-5 or iy < 5 or iy >= self.ny-5:
                continue

            # Простий розрахунок сили
            try:
                if self.use_gpu:
                    local_ux = float(self.ux[iy, ix])
                    local_uy = float(self.uy[iy, ix])
                else:
                    local_ux = self.ux[iy, ix]
                    local_uy = self.uy[iy, ix]

                # Проста сила опору
                drag = 0.01
                body.vx += (local_ux - body.vx) * drag
                body.vy += (local_uy - body.vy) * drag

                # Обмеження швидкості
                max_vel = 0.05
                vel_mag = math.sqrt(body.vx**2 + body.vy**2)
                if vel_mag > max_vel:
                    body.vx = body.vx * max_vel / vel_mag
                    body.vy = body.vy * max_vel / vel_mag

                # Оновлення позиції
                body.x += body.vx
                body.y += body.vy

                # Зіткнення зі стінками
                if body.x < body.radius:
                    body.x = body.radius
                    body.vx *= -0.5
                if body.x > self.nx - body.radius:
                    body.x = self.nx - body.radius
                    body.vx *= -0.5
                if body.y < body.radius:
                    body.y = body.radius
                    body.vy *= -0.5
                if body.y > self.ny - body.radius:
                    body.y = self.ny - body.radius
                    body.vy *= -0.5

            except Exception:
                # При помилці - зупинити тіло
                body.vx *= 0.9
                body.vy *= 0.9

    def reset_fields(self):
        """Скидання полів"""
        print("🔄 Скидання полів...")
        self.rho = self.xp.ones((self.ny, self.nx), dtype=self.xp.float32)
        self.ux = self.xp.ones((self.ny, self.nx), dtype=self.xp.float32) * self.u_inlet
        self.uy = self.xp.zeros((self.ny, self.nx), dtype=self.xp.float32)

        self.equilibrium()
        self.f[:] = self.feq[:]

    def add_moving_body(self, x, y, radius=20.0, color='red'):
        """Додати рухоме тіло"""
        body = MovingBody(x=x, y=y, radius=radius, color=color)
        self.moving_bodies.append(body)
        print(f"🔴 Додано {color} тіло")

    def clear_moving_bodies(self):
        """Очистити тіла"""
        self.moving_bodies.clear()

    def update_inlet_velocity(self, new_u_inlet):
        """Оновлення швидкості"""
        # Безпечне обмеження
        self.u_inlet = min(max(new_u_inlet, 0.01), 0.3)
        print(f"🔄 Швидкість: {new_u_inlet:.3f} → {self.u_inlet:.3f}")

    def get_velocity_magnitude(self):
        """Поле швидкості"""
        u_mag = self.xp.sqrt(self.ux**2 + self.uy**2)
        if self.use_gpu:
            return cp.asnumpy(u_mag)
        return u_mag

    def get_vorticity(self):
        """Завихреність"""
        if self.use_gpu:
            ux_cpu = cp.asnumpy(self.ux)
            uy_cpu = cp.asnumpy(self.uy)
        else:
            ux_cpu = self.ux
            uy_cpu = self.uy

        duy_dx = np.gradient(uy_cpu, axis=1)
        dux_dy = np.gradient(ux_cpu, axis=0)
        return duy_dx - dux_dy


class InteractiveLBM:
    """Простий інтерактивний інтерфейс"""

    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Простий стабільний LBM v10.0')

        # Параметри
        self.nx = 600
        self.ny = 200
        self.u_inlet = 0.1
        self.running = False
        self.adding_bodies = False

        # Створення симулятора
        self.lbm = SimpleLBM(self.nx, self.ny, self.u_inlet)

        # Інтерфейс
        self.setup_ui()
        self.setup_visualization()

        # Анімація
        self.animation = None
        self.body_patches = []

    def setup_ui(self):
        """Елементи керування"""
        # Основний графік
        self.ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        self.ax_main.set_aspect('equal')

        # Слайдер швидкості
        ax_vel = plt.subplot2grid((4, 4), (0, 3))
        self.slider_vel = Slider(ax_vel, 'Швидкість', 0.01, 0.3, valinit=self.u_inlet,
                                orientation='vertical')
        self.slider_vel.on_changed(self.update_velocity)

        # Кнопки
        ax_start = plt.subplot2grid((4, 4), (3, 0))
        self.btn_start = Button(ax_start, 'Старт')
        self.btn_start.on_clicked(self.toggle_simulation)

        ax_reset = plt.subplot2grid((4, 4), (3, 1))
        self.btn_reset = Button(ax_reset, 'Скидання')
        self.btn_reset.on_clicked(self.reset_simulation)

        ax_add = plt.subplot2grid((4, 4), (3, 2))
        self.btn_add = Button(ax_add, 'Додати тіло')
        self.btn_add.on_clicked(self.toggle_adding_bodies)

        ax_clear = plt.subplot2grid((4, 4), (3, 3))
        self.btn_clear = Button(ax_clear, 'Очистити')
        self.btn_clear.on_clicked(self.clear_bodies)

        # Обробка кліків
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def setup_visualization(self):
        """Візуалізація"""
        u_mag = self.lbm.get_velocity_magnitude()

        self.im = self.ax_main.imshow(
            u_mag,
            cmap='jet',
            origin='lower',
            vmin=0,
            vmax=0.3,
            interpolation='bilinear'
        )

        self.cbar = plt.colorbar(self.im, ax=self.ax_main)
        self.cbar.set_label('Швидкість')

        # Перешкода
        if self.lbm.use_gpu:
            obstacle = cp.asnumpy(self.lbm.obstacle)
        else:
            obstacle = self.lbm.obstacle
        self.ax_main.contour(obstacle, levels=[0.5], colors='black', linewidths=2)

        self.ax_main.set_title(f'Простий LBM: U = {self.u_inlet:.3f}')

    def on_click(self, event):
        """Додавання тіл"""
        if not self.adding_bodies or event.inaxes != self.ax_main:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        colors = ['red', 'blue', 'green', 'orange', 'purple']
        color = colors[len(self.lbm.moving_bodies) % len(colors)]

        self.lbm.add_moving_body(x, y, radius=15.0, color=color)

        circle = Circle((x, y), 15.0, color=color, alpha=0.7)
        self.ax_main.add_patch(circle)
        self.body_patches.append(circle)

        plt.draw()

    def update_velocity(self, val):
        """Оновлення швидкості"""
        self.u_inlet = val
        self.lbm.update_inlet_velocity(val)
        self.ax_main.set_title(f'Простий LBM: U = {self.u_inlet:.3f}')

    def toggle_adding_bodies(self, event):
        """Режим додавання тіл"""
        self.adding_bodies = not self.adding_bodies
        if self.adding_bodies:
            self.btn_add.label.set_text('Готово')
            print("👆 Клікніть для додавання тіла")
        else:
            self.btn_add.label.set_text('Додати тіло')

    def clear_bodies(self, event):
        """Очистити тіла"""
        self.lbm.clear_moving_bodies()

        for patch in self.body_patches:
            patch.remove()
        self.body_patches.clear()
        plt.draw()

    def toggle_simulation(self, event):
        """Старт/стоп"""
        if self.running:
            self.running = False
            self.btn_start.label.set_text('Старт')
            if self.animation:
                self.animation.event_source.stop()
        else:
            self.running = True
            self.btn_start.label.set_text('Стоп')
            if self.animation:
                self.animation.event_source.start()
            else:
                self.start_animation()

    def reset_simulation(self, event=None):
        """Скидання"""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
            self.btn_start.label.set_text('Старт')

        try:
            self.cbar.remove()
        except:
            pass

        for patch in self.body_patches:
            patch.remove()
        self.body_patches.clear()

        self.lbm = SimpleLBM(self.nx, self.ny, self.u_inlet)

        self.ax_main.clear()
        self.setup_visualization()
        plt.draw()

    def update_body_positions(self):
        """Оновлення позицій тіл"""
        for i, body in enumerate(self.lbm.moving_bodies):
            if i < len(self.body_patches):
                self.body_patches[i].center = (body.x, body.y)

    def update_frame(self, frame):
        """Оновлення кадру"""
        if not self.running:
            return [self.im]

        # Кроки симуляції
        for _ in range(5):
            self.lbm.step()

        # Оновлення
        self.update_body_positions()

        data = self.lbm.get_velocity_magnitude()
        self.im.set_array(data)
        self.im.set_clim(0, 0.3)

        return [self.im] + self.body_patches

    def start_animation(self):
        """Запуск анімації"""
        self.animation = animation.FuncAnimation(
            self.fig, self.update_frame,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

    def show(self):
        """Показати вікно"""
        plt.tight_layout()
        print("""
╔════════════════════════════════════════════╗
║    ПРОСТИЙ СТАБІЛЬНИЙ LBM v10.0            ║
║    🛡️ МАКСИМАЛЬНА СТАБІЛЬНІСТЬ            ║
║    🎯 БЕЗ СКЛАДНИХ РОЗРАХУНКІВ            ║
╚════════════════════════════════════════════╝

🛡️ ОСОБЛИВОСТІ:
✅ Фіксовані безпечні параметри (τ = 1.0)
✅ Простий стабільний FSI
✅ Автоматичне обмеження швидкості (0.01-0.3)
✅ Безпечна обробка помилок
✅ Відсутність складних розрахунків Re

🎮 ІНСТРУКЦІЇ:
1. Натисніть 'Додати тіло' і клікніть по полю
2. Натисніть 'Старт' для запуску
3. Регулюйте швидкість слайдером
4. Тіла будуть рухатися під дією потоку

🔧 ПАРАМЕТРИ:
- Швидкість: 0.01-0.3 (безпечний діапазон)
- τ = 1.0 (стабільне значення)
- Простий опір для тіл
        """)
        plt.show()


def main():
    """Головна функція"""
    print("""
╔════════════════════════════════════════════╗
║         ПРОСТИЙ LBM v10.0                  ║
║         🛡️ МАКСИМАЛЬНА СТАБІЛЬНІСТЬ       ║
║         🎯 ПРАЦЮЄ ЗАВЖДИ                   ║
╚════════════════════════════════════════════╝
    """)

    app = InteractiveLBM()
    app.show()


if __name__ == "__main__":
    main()