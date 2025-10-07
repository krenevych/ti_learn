import matplotlib
import numpy as np
import matplotlib.pyplot as plt


matplotlib.use("TkAgg")

# Параметри
nx = 61
dx = 2 / (nx - 1)
nt = 120
dt = 0.0125
c = 1.0

sigma = c * dt / dx
print("CFL =", sigma)  # бажано <= 1 для стабільності

# Сітка та початкова умова
x = np.linspace(0, 2, nx)
u = np.ones(nx)
u[int(0.5 / dx): int(1 / dx) + 1] = 2.0

# Інтерактивний режим оновлення графіка
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, u, lw=2)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0.5, 2.5)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('1D advection (upwind)')

un = u.copy()
for n in range(nt):
    un[:] = u
    for i in range(1, nx):
        u[i] = un[i] - sigma * (un[i] - un[i - 1])  # upwind
    # (необов’язково) гранична умова: залишаємо u[0] як є
    # якщо хочеш періодику, додай: u[0] = u[-1]

    line.set_ydata(u)
    fig.canvas.draw_idle()
    plt.pause(0.01)   # ключова коротка пауза для анімації

plt.ioff()
plt.show()
