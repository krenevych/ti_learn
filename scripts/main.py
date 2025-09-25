import taichi as ti

vec3 = ti.types.vector(3, ti.f64)

@ti.dataclass
class Ray:
    ro: vec3
    rd: vec3
    t: float

# The definition above is equivalent to
#Ray = ti.types.struct(ro=vec3, rd=vec3, t=float)
# Use positional arguments to set struct members in order
ray = Ray(vec3(1), vec3(111), 1.0)
print(ray)

# ro is set to vec3(0) and t will be set to 0
ray = Ray(vec3(0), rd=vec3(1, 0, 0))
print(ray)

# both ro and rd are set to vec3(0)
ray = Ray(t=1.0)
print(ray)

# ro is set to vec3(1), rd=vec3(0) and t=0.0
ray = Ray(1, t=1)
print(ray)

# All members are set to 0
ray = Ray()
print(ray)

pass
