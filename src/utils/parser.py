import taichi as ti

def backend(name):
    mapping = {
        "cpu": ti.cpu,
        "gpu": ti.gpu,
        "metal": ti.metal,
        "vulkan": ti.vulkan, "vk": ti.vulkan,
        "cuda": ti.cuda,
        "opengl": ti.opengl, "gl": ti.opengl,
    }

    return mapping[name]