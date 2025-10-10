import taichi as ti
import numpy as np
import time

# Initialize Taichi with GPU backend
# ti.init(arch=ti.gpu)
ti.init(arch=ti.opengl)

# Simulation parameters
Nx, Ny = 1600, 400  # Grid dimensions
Nt = 100000  # Number of time steps
tau = 0.6  # Relaxation time
cylinder_r = 33  # Cylinder radius

# D2Q9 lattice parameters
Q = 9
cxs = ti.Vector([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = ti.Vector([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = ti.Vector([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
opposite = ti.Vector([0, 5, 6, 7, 8, 1, 2, 3, 4])

# Taichi fields
F = ti.field(dtype=ti.f32, shape=(Nx, Ny, Q))
F_new = ti.field(dtype=ti.f32, shape=(Nx, Ny, Q))
rho = ti.field(dtype=ti.f32, shape=(Nx, Ny))
ux = ti.field(dtype=ti.f32, shape=(Nx, Ny))
uy = ti.field(dtype=ti.f32, shape=(Nx, Ny))
cylinder = ti.field(dtype=ti.i32, shape=(Nx, Ny))
vorticity = ti.field(dtype=ti.f32, shape=(Nx, Ny))

# Visualization field (RGB)
display_field = ti.Vector.field(3, dtype=ti.f32, shape=(Nx, Ny))

# Physical quantities for display
avg_velocity = ti.field(dtype=ti.f32, shape=())
max_velocity = ti.field(dtype=ti.f32, shape=())
avg_density = ti.field(dtype=ti.f32, shape=())
kinetic_energy = ti.field(dtype=ti.f32, shape=())
max_vorticity = ti.field(dtype=ti.f32, shape=())

@ti.kernel
def init_simulation(initial_velocity: ti.f32):
    """Initialize the simulation fields"""
    # Initialize cylinder (obstacle)
    cx, cy = Nx // 4, Ny // 2
    for i, j in ti.ndrange(Nx, Ny):
        if (i - cx) ** 2 + (j - cy) ** 2 <= cylinder_r ** 2:
            cylinder[i, j] = 1
        else:
            cylinder[i, j] = 0
    
    # Initialize distribution functions with equilibrium for uniform flow
    for i, j in ti.ndrange(Nx, Ny):
        if cylinder[i, j] == 0:
            rho_local = 1.0
            ux_local = initial_velocity
            uy_local = 0.0
            
            for q in ti.static(range(Q)):
                cu = cxs[q] * ux_local + cys[q] * uy_local
                F[i, j, q] = rho_local * weights[q] * (
                    1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux_local**2 + uy_local**2)
                )

@ti.kernel
def streaming():
    """Streaming step with periodic boundary conditions"""
    for i, j, q in ti.ndrange(Nx, Ny, Q):
        ip = (i + cxs[q]) % Nx
        jp = (j + cys[q]) % Ny
        F_new[ip, jp, q] = F[i, j, q]

@ti.kernel
def compute_macro():
    """Compute macroscopic quantities"""
    for i, j in ti.ndrange(Nx, Ny):
        rho[i, j] = 0.0
        ux[i, j] = 0.0
        uy[i, j] = 0.0
        
        for q in ti.static(range(Q)):
            rho[i, j] += F_new[i, j, q]
        
        for q in ti.static(range(Q)):
            ux[i, j] += F_new[i, j, q] * cxs[q]
            uy[i, j] += F_new[i, j, q] * cys[q]
        
        if rho[i, j] > 0:
            ux[i, j] /= rho[i, j]
            uy[i, j] /= rho[i, j]
        
        # Set velocity to zero inside cylinder
        if cylinder[i, j] == 1:
            ux[i, j] = 0.0
            uy[i, j] = 0.0

@ti.kernel
def collision(relaxation_tau: ti.f32):
    """Collision step with BGK operator"""
    for i, j in ti.ndrange(Nx, Ny):
        if cylinder[i, j] == 0:
            rho_local = rho[i, j]
            ux_local = ux[i, j]
            uy_local = uy[i, j]
            
            for q in ti.static(range(Q)):
                cu = cxs[q] * ux_local + cys[q] * uy_local
                F_eq = rho_local * weights[q] * (
                    1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux_local**2 + uy_local**2)
                )
                F[i, j, q] = F_new[i, j, q] - (1.0 / relaxation_tau) * (F_new[i, j, q] - F_eq)

@ti.kernel
def apply_boundary():
    """Apply bounce-back boundary condition on cylinder"""
    for i, j in ti.ndrange(Nx, Ny):
        if cylinder[i, j] == 1:
            for q in ti.static(range(Q)):
                F[i, j, q] = F_new[i, j, opposite[q]]

@ti.kernel
def compute_vorticity():
    """Compute vorticity field"""
    for i, j in ti.ndrange(Nx, Ny):
        if cylinder[i, j] == 0:
            ip = (i + 1) % Nx
            im = (i - 1) % Nx
            jp = (j + 1) % Ny
            jm = (j - 1) % Ny
            
            dux_dy = (ux[i, jp] - ux[i, jm]) / 2.0
            duy_dx = (uy[ip, j] - uy[im, j]) / 2.0
            vorticity[i, j] = duy_dx - dux_dy
        else:
            vorticity[i, j] = 0.0

@ti.kernel
def compute_statistics():
    """Compute physical quantities for display"""
    vel_sum = 0.0
    vel_max = 0.0
    rho_sum = 0.0
    ke_sum = 0.0
    vort_max = 0.0
    count = 0
    
    for i, j in ti.ndrange(Nx, Ny):
        if cylinder[i, j] == 0:
            vel = ti.sqrt(ux[i, j]**2 + uy[i, j]**2)
            vel_sum += vel
            vel_max = ti.max(vel_max, vel)
            rho_sum += rho[i, j]
            ke_sum += rho[i, j] * (ux[i, j]**2 + uy[i, j]**2) / 2.0
            vort_max = ti.max(vort_max, ti.abs(vorticity[i, j]))
            count += 1
    
    if count > 0:
        avg_velocity[None] = vel_sum / count
        avg_density[None] = rho_sum / count
        kinetic_energy[None] = ke_sum / count
    max_velocity[None] = vel_max
    max_vorticity[None] = vort_max

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
def prepare_display(mode: ti.i32, vmin: ti.f32, vmax: ti.f32):
    """Prepare visualization field based on mode with colormap"""
    for i, j in ti.ndrange(Nx, Ny):
        value = 0.0
        if mode == 0:  # Velocity magnitude
            value = ti.sqrt(ux[i, j]**2 + uy[i, j]**2)
        elif mode == 1:  # Vorticity
            value = vorticity[i, j]
        elif mode == 2:  # Density
            value = rho[i, j]
        elif mode == 3:  # Horizontal velocity
            value = ux[i, j]
        
        # Normalize to [0, 1] range
        normalized = 0.5
        if vmax > vmin:
            normalized = (value - vmin) / (vmax - vmin)
            normalized = ti.max(0.0, ti.min(1.0, normalized))
        
        # Apply colormap
        if cylinder[i, j] == 1:
            # Mark cylinder as dark grey
            display_field[i, j] = ti.math.vec3(0.2, 0.2, 0.2)
        else:
            # Use rainbow colormap for better visibility
            display_field[i, j] = colormap_rainbow(normalized)

def main():
    """Main simulation loop with real-time visualization"""
    # Simulation control parameters
    initial_velocity = 0.1
    current_tau = tau
    paused = False
    
    print("Initializing LBM simulation with Taichi...")
    init_simulation(initial_velocity)
    
    # Create window for visualization
    window = ti.ui.Window("LBM Simulation - Real-Time", (1200, 400), vsync=False)
    canvas = window.get_canvas()
    
    frame_count = 0
    start_time = time.time()
    vis_mode = 1  # 0: velocity, 1: vorticity, 2: density, 3: ux
    mode_names = ["Velocity Magnitude", "Vorticity", "Density", "Horizontal Velocity"]
    
    print("Starting simulation...")
    print("Controls:")
    print("  1 - Velocity magnitude view")
    print("  2 - Vorticity view")
    print("  3 - Density view")
    print("  4 - Horizontal velocity view")
    print("  SPACE - Pause/Resume")
    print("  R - Reset simulation")
    print("  ESC - Exit")
    
    while window.running:
        # Handle keyboard input
        if window.get_event(ti.ui.PRESS):
            if window.event.key == '1':
                vis_mode = 0
            elif window.event.key == '2':
                vis_mode = 1
            elif window.event.key == '3':
                vis_mode = 2
            elif window.event.key == '4':
                vis_mode = 3
            elif window.event.key == ti.ui.SPACE:
                paused = not paused
                print(f"Simulation {'paused' if paused else 'resumed'}")
            elif window.event.key == 'r' or window.event.key == 'R':
                print("Resetting simulation...")
                init_simulation(initial_velocity)
                frame_count = 0
                start_time = time.time()
            elif window.event.key == ti.ui.ESCAPE:
                break
        
        # Perform multiple simulation steps per frame for speed (only if not paused)
        if not paused:
            for _ in range(10):
                streaming()
                compute_macro()
                collision(current_tau)
                apply_boundary()
                frame_count += 1
        
        # Update visualization every frame
        compute_vorticity()
        compute_statistics()
        
        # Set visualization range based on mode
        if vis_mode == 0:  # Velocity magnitude
            vmin, vmax = 0.0, 0.15
        elif vis_mode == 1:  # Vorticity
            vmin, vmax = -0.02, 0.02
        elif vis_mode == 2:  # Density
            vmin, vmax = 0.95, 1.05
        elif vis_mode == 3:  # Horizontal velocity
            vmin, vmax = 0.0, 0.15
        
        prepare_display(vis_mode, vmin, vmax)
        
        # Display the field
        canvas.set_image(display_field)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Display statistics and controls
        window.GUI.begin("Physical Quantities", 0.02, 0.02, 0.35, 0.65)
        window.GUI.text(f"Step: {frame_count}")
        window.GUI.text(f"FPS: {fps:.1f}")
        window.GUI.text(f"Status: {'PAUSED' if paused else 'Running'}")
        window.GUI.text(f"Mode: {mode_names[vis_mode]}")
        window.GUI.text("")
        
        window.GUI.text("Physical Quantities:")
        window.GUI.text(f"Avg Velocity: {avg_velocity[None]:.6f}")
        window.GUI.text(f"Max Velocity: {max_velocity[None]:.6f}")
        window.GUI.text(f"Avg Density: {avg_density[None]:.6f}")
        window.GUI.text(f"Kinetic Energy: {kinetic_energy[None]:.6f}")
        window.GUI.text(f"Max Vorticity: {max_vorticity[None]:.6f}")
        reynolds = initial_velocity * 2 * cylinder_r / ((current_tau - 0.5) / 3)
        window.GUI.text(f"Reynolds number: {reynolds:.1f}")
        window.GUI.text("")
        
        window.GUI.text("Simulation Parameters:")
        new_velocity = window.GUI.slider_float("Initial Velocity", initial_velocity, 0.01, 0.3)
        if abs(new_velocity - initial_velocity) > 0.001:
            initial_velocity = new_velocity
        
        new_tau = window.GUI.slider_float("Relaxation Time (tau)", current_tau, 0.51, 2.0)
        if abs(new_tau - current_tau) > 0.001:
            current_tau = new_tau
        
        window.GUI.text("")
        if window.GUI.button("Reset Simulation"):
            print(f"Resetting with velocity={initial_velocity:.3f}, tau={current_tau:.3f}")
            init_simulation(initial_velocity)
            frame_count = 0
            start_time = time.time()
        
        if window.GUI.button("Pause/Resume"):
            paused = not paused
        
        window.GUI.end()
        
        window.show()
    
    print(f"\nSimulation finished!")
    print(f"Total steps: {frame_count}")
    print(f"Average FPS: {fps:.1f}")

if __name__ == "__main__":
    main()
