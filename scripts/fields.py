import taichi as ti

ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu)


f_2d = ti.field(ti.f32, shape=(16, 16))

@ti.kernel
def loop_over_2d():
  for i, j in f_2d:
      f_2d[i, j] = i

loop_over_2d()

print(f_2d)