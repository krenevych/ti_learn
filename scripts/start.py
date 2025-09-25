import taichi as ti

from src.utils.parser import backend
from src.utils.reader import load_yaml

# load configuration

cfg = load_yaml("../configs/cloth_params.yaml")
window_resolution = tuple(cfg["window"]["resolution"])
background_color = tuple(cfg["canvas"]["background_color"])

ti.init(arch=backend(cfg["backend"]))


window = ti.ui.Window("Taichi Cloth Simulation on GGUI",
                      window_resolution,  # (1024, 1024),
                      vsync=True)

# setup canvas
canvas = window.get_canvas()
canvas.set_background_color(background_color)
scene = ti.ui.Scene()
camera = ti.ui.Camera()

def update_camera(scene, camera):
    camera.position(*cfg["camera"]["position"])
    camera.lookat(*cfg["camera"]["lookat"])
    scene.set_camera(camera)

def update_light(scene):
    scene.point_light(
        pos=tuple(cfg["light"]["point"]["position"]),
        color=tuple(cfg["light"]["point"]["color"])
    )

    scene.ambient_light(cfg["light"]["ambient"]["color"])

gravity = ti.Vector(cfg["gravity"])
dt = cfg["time_step"]
dashpot_damping = cfg["dashpot_damping"]
drag_damping = cfg["drag_damping"]

# Ball configuration
ball = cfg["ball"]
ball_radius = ball["radius"]
ball_color  = tuple(ball["color"])
ball_centers = ball["positions"]
ball_number = len(ball_centers)
ball_positions = ti.Vector.field(3, dtype=ti.f32, shape=(ball_number,))
ball_velocities = ti.Vector.field(3, dtype=ti.f32, shape=(ball_number, ))
for i in range(ball_number):
    ball_positions[i] = ball_centers[i]
    ball_velocities[i] = [0, 0, 0]

x0, y0 = 0.0, 0.0
x1, y1 = 1.0, 0.0

@ti.kernel
def update():
    for i in ball_velocities:
        ball_velocities[i] += gravity * dt #/ m

    for i in ball_positions:
        if ball_positions[i].y <= 0.0:
            ball_positions[i].y = 0.0
            # normal = ti.Vector([0.0, 1.0, 0.0])
            # ball_velocities[i] -= min(ball_velocities[i].dot(normal), 0) * normal
            # ball_velocities[i] *= drag_damping
            ball_velocities[i] = -ball_velocities[i]#.dot(normal) * normal


    for i in ball_positions:
        ball_positions[i] += dt * ball_velocities[i]



def update_meshes(scene):
    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_positions, radius=ball_radius * 0.95, color=ball_color)

current_t = 0.0

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

    update()

    update_meshes(scene)
    update_light(scene)
    update_camera(scene, camera)

    # gui.line([x0, y0], [x1, y0], color=0xEEEEF0, radius=2)  # низ
    # canvas.lines([x0, y0], [x1, y0])

    canvas.scene(scene)
    window.show()

    current_t += dt


