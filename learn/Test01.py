import taichi as ti
import numpy as np
import time

# Initialize Taichi with GPU backend
# ti.init(arch=ti.gpu)
ti.init(arch=ti.opengl)

# Simulation parameters
Nx, Ny = 1600, 400  # Grid dimensions
Nt = 100000         # Number of time steps
tau = 0.6           # Relaxation time
cylinder_r = 33     # Cylinder radius

initial_velocity = 0.1

cs2 = 1.0 / 3.0     # c_s^2
cs4 = cs2 * cs2     # c_s^4 = (1/3)^2 = 1/9

# D2Q9 lattice parameters
Q = 9
#                     0  1  2  3   4   5   6   7   8
cxs      = ti.Vector([0, 0, 1, 1,  1,  0, -1, -1, -1])
cys      = ti.Vector([0, 1, 1, 0, -1, -1, -1,  0,  1])
opposite = ti.Vector([0, 5, 6, 7,  8,  1,  2,  3,  4])
w        = ti.Vector([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

# Taichi fields
f         = ti.field(dtype=ti.f32, shape=(Nx, Ny, Q))
f_new     = ti.field(dtype=ti.f32, shape=(Nx, Ny, Q))
rho       = ti.field(dtype=ti.f32, shape=(Nx, Ny))
ux        = ti.field(dtype=ti.f32, shape=(Nx, Ny))
uy        = ti.field(dtype=ti.f32, shape=(Nx, Ny))
cylinder  = ti.field(dtype=ti.i32, shape=(Nx, Ny))
vorticity = ti.field(dtype=ti.f32, shape=(Nx, Ny))

# Visualization field (RGB)
display_field = ti.Vector.field(3, dtype=ti.f32, shape=(Nx, Ny))

@ti.func
def mark_obstacle():
    # Initialize cylinder (obstacle)
    cx, cy = Nx // 4, Ny // 2
    cx1, cy1 = 0.8 * Nx // 4, 2.8 * Ny // 4
    for i, j in ti.ndrange(Nx, Ny):
        if (i - cx) ** 2 + (j - cy) ** 2 <= cylinder_r ** 2:
            cylinder[i, j] = 1
        elif (i - cx1) ** 2 + (j - cy1) ** 2 <= cylinder_r ** 2:
            cylinder[i, j] = 1
        else:
            cylinder[i, j] = 0

@ti.func
def is_obstacle(i, j):
    return cylinder[i, j] == 1

@ti.kernel
def init_simulation(initial_velocity: ti.f32):
    """Initialize the simulation fields"""
    mark_obstacle()

    # Initialize distribution functions with equilibrium for uniform flow
    for i, j in ti.ndrange(Nx, Ny):
        if not is_obstacle(i, j):
            rho_local = 1.0
            ux_local = initial_velocity
            uy_local = 0.0
            dot_u = ux_local ** 2 + uy_local ** 2

            for q in ti.static(range(Q)):
                cu = cxs[q] * ux_local + cys[q] * uy_local
                f[i, j, q] = rho_local * w[q] * (1.0 + cu / cs2 + 0.5 * cu * cu / cs4 - 0.5 * dot_u / cs2)


@ti.kernel
def streaming():
    """Streaming step with periodic boundary conditions"""
    for i, j, q in ti.ndrange(Nx, Ny, Q):
        ip = (i + cxs[q]) % Nx
        jp = (j + cys[q]) % Ny
        f_new[ip, jp, q] = f[i, j, q]

@ti.kernel
def compute_macro():
    """Compute macroscopic quantities"""
    for i, j in ti.ndrange(Nx, Ny):
        rho[i, j] = 0.0
        ux[i, j] = 0.0
        uy[i, j] = 0.0

        for q in ti.static(range(Q)):
            rho[i, j] += f_new[i, j, q]

        for q in ti.static(range(Q)):
            ux[i, j] += f_new[i, j, q] * cxs[q]
            uy[i, j] += f_new[i, j, q] * cys[q]

        if rho[i, j] > 0:
            ux[i, j] /= rho[i, j]
            uy[i, j] /= rho[i, j]

        if is_obstacle(i, j):  # Set velocity to zero inside cylinder
            ux[i, j] = 0.0
            uy[i, j] = 0.0

@ti.kernel
def collision(relaxation_tau: ti.f32):
    """Collision step with BGK operator"""
    for i, j in ti.ndrange(Nx, Ny):
        if not is_obstacle(i, j):
            rho_local = rho[i, j]
            ux_local = ux[i, j]
            uy_local = uy[i, j]
            dot_u = ux_local ** 2 + uy_local ** 2

            for q in ti.static(range(Q)):
                cu = cxs[q] * ux_local + cys[q] * uy_local  # cu = dot(c_i, u)
                f_eq = rho_local * w[q] * (1.0 + cu / cs2 + 0.5 * cu * cu / cs4 - 0.5 * dot_u / cs2)
                f[i, j, q] = f_new[i, j, q] - (1.0 / relaxation_tau) * (f_new[i, j, q] - f_eq)

@ti.kernel
def apply_boundary():
    """Apply bounce-back boundary condition on cylinder"""
    for i, j in ti.ndrange(Nx, Ny):
        if is_obstacle(i, j):
            for q in ti.static(range(Q)):
                f[i, j, q] = f_new[i, j, opposite[q]]

@ti.kernel
def compute_vorticity():
    """Compute vorticity field"""
    for i, j in ti.ndrange(Nx, Ny):
        if is_obstacle(i, j):
            vorticity[i, j] = 0.0
        else:
            ip = (i + 1) % Nx
            im = (i - 1) % Nx
            jp = (j + 1) % Ny
            jm = (j - 1) % Ny

            dux_dy = (ux[i, jp] - ux[i, jm]) / 2.0
            duy_dx = (uy[ip, j] - uy[im, j]) / 2.0
            vorticity[i, j] = duy_dx - dux_dy


@ti.func
def colormap_jet(value: ti.f32) -> ti.math.vec3:
    """Jet colormap: blue -> cyan -> green -> yellow -> red"""
    r, g, b = 0.0, 0.0, 0.0

    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0

    if value < 0.25:
        b = 1.0
        g = value / 0.25
    elif value < 0.5:
        g = 1.0
        b = 1.0 - (value - 0.25) / 0.25
    elif value < 0.75:
        g = 1.0
        r = (value - 0.5) / 0.25
    else:
        r = 1.0
        g = 1.0 - (value - 0.75) / 0.25

    return ti.math.vec3(r, g, b)

@ti.func
def colormap_rainbow(value: ti.f32) -> ti.math.vec3:
    """Rainbow colormap: violet -> blue -> green -> yellow -> red"""
    r, g, b = 0.0, 0.0, 0.0

    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0

    if value < 0.2:
        r = (0.2 - value) / 0.2 * 0.5
        b = 1.0
    elif value < 0.4:
        g = (value - 0.2) / 0.2
        b = 1.0
    elif value < 0.6:
        g = 1.0
        b = 1.0 - (value - 0.4) / 0.2
    elif value < 0.8:
        r = (value - 0.6) / 0.2
        g = 1.0
    else:
        r = 1.0
        g = 1.0 - (value - 0.8) / 0.2

    return ti.math.vec3(r, g, b)

@ti.kernel
def prepare_display(vmin: ti.f32, vmax: ti.f32):
    """Prepare visualization field based on mode with colormap"""
    for i, j in ti.ndrange(Nx, Ny):
        value = vorticity[i, j]

        # Normalize to [0, 1] range
        normalized = 0.5
        if vmax > vmin:
            normalized = (value - vmin) / (vmax - vmin)
            normalized = ti.max(0.0, ti.min(1.0, normalized))

        # Apply colormap
        if is_obstacle(i, j):
            display_field[i, j] = ti.math.vec3(0.2, 0.2, 0.2)  # Mark cylinder as dark grey
        else:
            display_field[i, j] = colormap_rainbow(normalized)  # Use rainbow colormap for better visibility

def main():
    """Main simulation loop with real-time visualization"""
    print("Initializing LBM simulation with Taichi...")
    init_simulation(initial_velocity)

    # Create window for visualization
    window = ti.ui.Window("LBM Simulation - Real-Time", (1200, 400), vsync=False)
    canvas = window.get_canvas()

    frame_count = 0
    print("Starting simulation...")
    print("Controls:")
    print("  R - Reset simulation")
    print("  ESC - Exit")

    while window.running:
        # Handle keyboard input
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r' or window.event.key == 'R':
                print("Resetting simulation...")
                init_simulation(initial_velocity)
                frame_count = 0
            elif window.event.key == ti.ui.ESCAPE:
                break

        # Perform multiple simulation steps per frame for speed (only if not paused)
        for _ in range(10):
            streaming()
            compute_macro()
            collision(tau)
            apply_boundary()
            frame_count += 1

        # Update visualization every frame
        compute_vorticity()

        vmin, vmax = -0.02, 0.02
        prepare_display(vmin, vmax)

        # Display the field
        canvas.set_image(display_field)
        window.show()

    print(f"\nSimulation finished!")

if __name__ == "__main__":
    main()
