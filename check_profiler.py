import taichi as ti, time
ti.init(arch=ti.vulkan, kernel_profiler=True)
print("[arch]", ti.lang.impl.current_cfg().arch, "Taichi", ti.__version__)

N = 500_000
a = ti.field(ti.f32, shape=N)
s = ti.field(ti.f32, shape=())

@ti.kernel
def fill_lin():
    ti.loop_config(block_dim=256)   # підказка GPU (не обов’язково)
    for i in range(N):
        a[i] = 0.001 * i            # дешева арифметика замість sin

@ti.kernel
def reduce_sum():
    acc = 0.0
    for i in range(N):
        acc += a[i]
    s[None] = acc                   # побічний ефект => ядро точно «рахується»

# warm-up (JIT + перші проггони)
fill_lin(); reduce_sum(); ti.sync()

t0 = time.perf_counter()
for _ in range(5):
    fill_lin()
    reduce_sum()
ti.sync()
t1 = time.perf_counter()

print("checksum:", s[None], "elapsed:", round(t1 - t0, 3), "s")
ti.profiler.print_kernel_profiler_info()
ti.profiler.clear_kernel_profiler_info()
