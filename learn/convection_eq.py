# Remember: comments in python are denoted by the pound sign
import matplotlib
import numpy as np                 #here we load numpy and create a shortcut np
from matplotlib import pyplot      #here we load matplotlib
import time, sys                   #and load some utilities

#this makes matplotlib plots appear in the notebook (instead of a separate window)
# %matplotlib inline

matplotlib.use("TkAgg")

nx = 61 # try changing this number from 41 to 81 and Run All... What happens?
dx = 2 / (nx-1)  # delta x
print(dx)

nt = 20     #nt is the number of timesteps we want to calculate
dt = 0.025  #dt is the number of time that each timestep covers (delta t)
c = 1     #assume wavespeed of c = 1

u = np.ones(nx)      #numpy function ones()
u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s
print(u)
# print(np.linspace(0, 2, nx))

# pyplot.plot(np.linspace(0, 2, nx), u)
# pyplot.show()

un = np.ones(nx) # initialize a temporary array

for n in range(nt): # loop n from 0 to nt-1, so it will run nt times
    un = u.copy() # copy the existing values of u into un
    for i in range(1, nx): # change the range to 0 to nx, and see what happens
        u[i] = un[i] - c * dt / dx * (un[i]-un[i-1])


    pyplot.plot(np.linspace(0, 2, nx), u)

pyplot.show()
