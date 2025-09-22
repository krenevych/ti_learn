import taichi as ti

ti.init(arch=ti.gpu)


@ti.kernel
def test_kernel(x: int):
    print(x)


test_kernel(42)
