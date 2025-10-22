# We'll create a simplified, minimal LBM (D2Q9) cavity-flow demo in Taichi.
# - Single-file script that runs on launch
# - Defaults everywhere
# - No GUI controls, no stats, no extra visualizations
# - Square domain, lid-driven cavity (moving top wall), bounce-back boundaries
# - Periodic in x is NOT used; all walls are solid; only top wall moves
# - Uses BGK collision, pull-streaming, grayscale velocity magnitude render

import taichi as ti
ti.init(arch=ti.opengl)  # default to opengl for widest compatibility

# -------- Parameters (defaults) --------
Nx, Ny = 256, 256
tau = 0.6            # relaxation time (BGK)
U_lid = 0.1          # top wall horizontal speed
substeps_per_frame = 10

# D2Q9
Q = 9
c = [
    (0, 0),
    (1, 0), (0, 1), (-1, 0), (0, -1),
    (1, 1), (-1, 1), (-1, -1), (1, -1),
]
w = [4/9,
     1/9, 1/9, 1/9, 1/9,
     1/36, 1/36, 1/36, 1/36]
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # opposite directions
cs2 = 1/3

# -------- Fields --------
f     = ti.field(dtype=ti.f32, shape=(Nx, Ny, Q))
f_new = ti.field(dtype=ti.f32, shape=(Nx, Ny, Q))
rho   = ti.field(dtype=ti.f32, shape=(Nx, Ny))
ux    = ti.field(dtype=ti.f32, shape=(Nx, Ny))
uy    = ti.field(dtype=ti.f32, shape=(Nx, Ny))
img   = ti.field(dtype=ti.f32, shape=(Nx, Ny))  # grayscale velocity magnitude

# -------- Helpers --------
@ti.func
def feq(q: ti.i32, rho_v: ti.f32, ux_v: ti.f32, uy_v: ti.f32) -> ti.f32:
    cx, cy = c[q]
    cu = cx * ux_v + cy * uy_v
    u2 = ux_v * ux_v + uy_v * uy_v
    return w[q] * rho_v * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)

@ti.kernel
def initialize():
    for i, j in ti.ndrange(Nx, Ny):
        rho[i, j] = 1.0
        ux[i, j] = 0.0
        uy[i, j] = 0.0
        for q in range(Q):
            f[i, j, q] = feq(q, rho[i, j], ux[i, j], uy[i, j])

@ti.kernel
def collide_and_stream():
    # Pull-streaming: read neighbors, then collide in-place into f_new
    for i, j in ti.ndrange(Nx, Ny):
        # Pull step: gather populations for this cell from neighbor cells
        # after streaming; here we read from f (previous step) at neighbor positions
        fin = ti.Vector.zero(ti.f32, Q)
        for q in range(Q):
            cx, cy = c[q]
            ip = i - cx
            jp = j - cy
            # clamp to domain (solid walls): we will handle walls via bounce-back later
            if ip < 0:   ip = 0
            if ip >= Nx: ip = Nx - 1
            if jp < 0:   jp = 0
            if jp >= Ny: jp = Ny - 1
            fin[q] = f[ip, jp, q]

        # Macros
        rho_ = 0.0
        ux_ = 0.0
        uy_ = 0.0
        for q in range(Q):
            rho_ += fin[q]
        for q in range(Q):
            cx, cy = c[q]
            ux_ += fin[q] * cx
            uy_ += fin[q] * cy
        ux_ /= rho_
        uy_ /= rho_

        # Collision (BGK)
        for q in range(Q):
            feq_ = feq(q, rho_, ux_, uy_)
            f_new[i, j, q] = fin[q] - (1.0 / tau) * (fin[q] - feq_)

@ti.kernel
def apply_bounce_back_and_moving_lid():
    # Simple halfway bounce-back at all walls.
    # For the top lid (j = Ny-1), apply moving-wall correction with velocity (U_lid, 0).
    for i in range(Nx):
        # Bottom wall (j=0)
        j = 0
        # Directions pointing into the fluid from the wall normal:
        # Bottom wall normal is (0, -1), so bounce pairs are: q2<->q4 (up<->down), q5<->q7, q8<->q6
        f_new[i, j, 2] = f_new[i, j, 4]  # up    <- down
        f_new[i, j, 5] = f_new[i, j, 7]  # up-right <- down-left
        f_new[i, j, 8] = f_new[i, j, 6]  # up-left  <- down-right

        # Top wall (j=Ny-1) with moving lid (u = (U_lid, 0))
        j = Ny - 1
        rho_top = rho[i, j]
        # Basic moving-wall bounce-back correction:
        # f_i = f_opp - 6*w_i*rho*(c_i·u_w)
        # Affects directions coming from the wall into the fluid (downward-pointing):
        # q4 (down) <- q2 (up), q7 (down-left) <- q5 (up-right), q6 (down-right) <- q8 (up-left)
        # Note: c_i for q4 is (0,-1), q7 is (-1,-1), q6 is (1,-1)
        f_new[i, j, 4] = f_new[i, j, 2] - 6.0 * w[4] * rho_top * ((0 * U_lid) + (-1) * 0.0)
        f_new[i, j, 7] = f_new[i, j, 5] - 6.0 * w[7] * rho_top * ((-1) * U_lid + (-1) * 0.0)
        f_new[i, j, 6] = f_new[i, j, 8] - 6.0 * w[6] * rho_top * ((1) * U_lid + (-1) * 0.0)

    for j in range(Ny):
        # Left wall (i=0)
        i = 0
        # Left wall normal is (-1,0): bounce pairs q1<->q3, q5<->q6, q8<->q7
        f_new[i, j, 1] = f_new[i, j, 3]  # right <- left
        f_new[i, j, 5] = f_new[i, j, 6]  # up-right <- up-left
        f_new[i, j, 8] = f_new[i, j, 7]  # down-right <- down-left

        # Right wall (i=Nx-1)
        i = Nx - 1
        # Right wall normal is (1,0): bounce pairs q3<->q1, q6<->q5, q7<->q8
        f_new[i, j, 3] = f_new[i, j, 1]  # left <- right
        f_new[i, j, 6] = f_new[i, j, 5]  # up-left <- up-right
        f_new[i, j, 7] = f_new[i, j, 8]  # down-left <- down-right

@ti.kernel
def update_macros_and_swap():
    for i, j in ti.ndrange(Nx, Ny):
        # swap f <- f_new and compute macros
        rho_ = 0.0
        ux_ = 0.0
        uy_ = 0.0
        for q in range(Q):
            f[i, j, q] = f_new[i, j, q]
            rho_ += f[i, j, q]
        for q in range(Q):
            cx, cy = c[q]
            ux_ += f[i, j, q] * cx
            uy_ += f[i, j, q] * cy
        rho[i, j] = rho_
        ux[i, j] = ux_ / rho_
        uy[i, j] = uy_ / rho_

@ti.kernel
def render():
    vmax = 0.0
    # Find max |u| for normalization
    for i, j in ti.ndrange(Nx, Ny):
        speed = ti.sqrt(ux[i, j] * ux[i, j] + uy[i, j] * uy[i, j])
        vmax = ti.max(vmax, speed)
    if vmax < 1e-6:
        vmax = 1e-6
    for i, j in ti.ndrange(Nx, Ny):
        speed = ti.sqrt(ux[i, j] * ux[i, j] + uy[i, j] * uy[i, j])
        img[i, j] = speed / vmax  # grayscale [0,1]

def main():
    initialize()
    window = ti.ui.Window("Minimal LBM (D2Q9) - Lid-Driven Cavity", (Nx, Ny), vsync=True)
    canvas = window.get_canvas()

    while window.running:
        for _ in range(substeps_per_frame):
            collide_and_stream()
            apply_bounce_back_and_moving_lid()
            update_macros_and_swap()
        render()
        canvas.set_image(img)
        window.show()

if __name__ == "__main__":
    main()
