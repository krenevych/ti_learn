import taichi as ti

# Initialize Taichi
ti.init(arch=ti.gpu)

# Simulation parameters
Nx, Ny = 800, 200
tau = 0.6
cylinder_r = 20

# D2Q9 lattice parameters
Q = 9
cxs = ti.Vector([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = ti.Vector([0, 1, 1, 0, -1, -1, -1, 0, 1])
opposite = ti.Vector([0, 5, 6, 7, 8, 1, 2, 3, 4])
weights = ti.Vector([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

# Fields
f = ti.field(dtype=ti.f32, shape=(Nx, Ny, Q))
f_new = ti.field(dtype=ti.f32, shape=(Nx, Ny, Q))
rho = ti.field(dtype=ti.f32, shape=(Nx, Ny))
ux = ti.field(dtype=ti.f32, shape=(Nx, Ny))
uy = ti.field(dtype=ti.f32, shape=(Nx, Ny))
cylinder = ti.field(dtype=ti.i32, shape=(Nx, Ny))
display = ti.Vector.field(3, dtype=ti.f32, shape=(Nx, Ny))

@ti.kernel
def init():
    # Initialize cylinder obstacle
    cx, cy = Nx // 4, Ny // 2
    for i, j in ti.ndrange(Nx, Ny):
        cylinder[i, j] = 1 if (i - cx)**2 + (j - cy)**2 <= cylinder_r**2 else 0

    # Initialize flow
    for i, j in ti.ndrange(Nx, Ny):
        if cylinder[i, j] == 0:
            u0 = 0.1
            for q in ti.static(range(Q)):
                cu = cxs[q] * u0
                f[i, j, q] = weights[q] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u0*u0)

@ti.kernel
def streaming():
    for i, j, q in ti.ndrange(Nx, Ny, Q):
        ip = (i + cxs[q]) % Nx
        jp = (j + cys[q]) % Ny
        f_new[ip, jp, q] = f[i, j, q]

@ti.kernel
def collision():
    for i, j in ti.ndrange(Nx, Ny):
        # Compute macroscopic quantities
        rho[i, j] = 0.0
        ux[i, j] = 0.0
        uy[i, j] = 0.0

        for q in ti.static(range(Q)):
            rho[i, j] += f_new[i, j, q]
            ux[i, j] += f_new[i, j, q] * cxs[q]
            uy[i, j] += f_new[i, j, q] * cys[q]

        if rho[i, j] > 0:
            ux[i, j] /= rho[i, j]
            uy[i, j] /= rho[i, j]

        # BGK collision
        if cylinder[i, j] == 0:
            for q in ti.static(range(Q)):
                cu = cxs[q] * ux[i, j] + cys[q] * uy[i, j]
                feq = rho[i, j] * weights[q] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(ux[i, j]**2 + uy[i, j]**2))
                f[i, j, q] = f_new[i, j, q] - (f_new[i, j, q] - feq) / tau
        else:
            # Bounce-back on cylinder
            for q in ti.static(range(Q)):
                f[i, j, q] = f_new[i, j, opposite[q]]

@ti.kernel
def visualize():
    for i, j in ti.ndrange(Nx, Ny):
        if cylinder[i, j] == 1:
            display[i, j] = ti.Vector([0.2, 0.2, 0.2])
        else:
            v = ti.sqrt(ux[i, j]**2 + uy[i, j]**2)
            c = ti.min(v * 10.0, 1.0)
            display[i, j] = ti.Vector([c, c * 0.5, 1.0 - c])

def main():
    init()
    window = ti.ui.Window("LBM Simulation", (Nx, Ny))
    canvas = window.get_canvas()

    while window.running:
        for _ in range(10):
            streaming()
            collision()

        visualize()
        canvas.set_image(display)
        window.show()

if __name__ == "__main__":
    main()
