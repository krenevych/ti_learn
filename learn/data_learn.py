# import taichi as ti
#
# ti.init(arch=ti.cpu)   # CPU fallback
#
# transform_type = ti.types.struct(R=ti.math.mat3, T=ti.math.vec3)
# pos_type = ti.types.struct(x=ti.math.vec3, trans=transform_type)
#
# @ti.kernel
# def kernel_with_nested_struct_arg(p: pos_type) -> ti.math.vec3:
#     return p.trans.R @ p.x + p.trans.T
#
# trans = transform_type(ti.math.mat3(1), [1, 1, 1])
# p = pos_type(x=[1, 1, 1], trans=trans)
# print(kernel_with_nested_struct_arg(p))  # [4., 4., 4.]


import taichi as ti
ti.init()

a = 1

@ti.kernel
def kernel_1():
    print(a)

@ti.kernel
def kernel_2():
    print(a)

kernel_1()  # Prints 1
a = 2
kernel_1()  # Prints 1
kernel_2()  # Prints 2