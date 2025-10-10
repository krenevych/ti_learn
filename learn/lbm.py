import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

# LBM Parameters
Nx = 400  # resolution x-dir
Ny = 100  # resolution y-dir
rho0 = 100  # average density
tau = 0.6  # collision timescale
Nt = 4000  # number of timesteps

# Lattice speeds / weights
NL = 9
idxs = np.arange(NL)
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])  # sums to 1

# Initial Conditions
F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * np.arange(Nx) / Nx * 4))
rho = np.sum(F, 2)
for i in idxs:
    F[:, :, i] *= rho0 / rho

# Cylinder boundary
X, Y = np.meshgrid(range(Nx), range(Ny))
cylinder = (X - Nx//4)**2 + (Y - Ny//2)**2 < (Ny//4)**2

# Initialize a list to store the frames
frames = []

for it in range(Nt):
    # Drift (Streaming)
    for i, cx, cy in zip(idxs, cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

    # Set reflective boundaries
    bndryF = F[cylinder, :]
    bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

    # Calculate fluid variables
    rho = np.sum(F, 2)
    ux = np.sum(F * cxs, 2) / rho
    uy = np.sum(F * cys, 2) / rho

    # Apply Collision
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy)
                                  + 9 * (cx * ux + cy * uy) ** 2 / 2 - 3 * (ux ** 2 + uy ** 2) / 2)

    F += -(1.0 / tau) * (F - Feq)

    # Apply boundary
    F[cylinder, :] = bndryF

    # Plot in real time
    if ((it % 50) == 0) or (it == Nt - 1):
        print(f"Timestep: {it}/{Nt}")
        plt.cla()  # Clear the current axes
        ux[cylinder] = 0
        uy[cylinder] = 0
        # Calculate vorticity
        vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                    np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        vorticity[cylinder] = np.nan  # Set vorticity inside the cylinder to NaN
        # Create a masked array to avoid plotting cylinder area
        vorticity_masked = np.ma.array(vorticity, mask=cylinder)
        # Plot vorticity
        plt.imshow(vorticity_masked, cmap='bwr')
        plt.colorbar()
        plt.clim(-.1, .1)  # Set the color limits
        # Plot cylinder
        plt.imshow(~cylinder, cmap='gray', alpha=0.3)
        # Set axis properties
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

        # Save the current frame
        plt.savefig(f'frame_{it}.png', dpi=240)
        frames.append(imageio.imread(f'frame_{it}.png'))
        plt.close()

# Create GIF
print("Creating GIF...")
imageio.mimsave('lbm_flow.gif', frames, fps=10)
print("Done! GIF saved as 'lbm_flow.gif'")

# Clean up frame files
for it in range(0, Nt, 50):
    if os.path.exists(f'frame_{it}.png'):
        os.remove(f'frame_{it}.png')
if os.path.exists(f'frame_{Nt-1}.png'):
    os.remove(f'frame_{Nt-1}.png')
