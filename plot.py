import numpy as np
import matplotlib.pyplot as plt

for filename in ("V1.dat", "source.dat"):
    data = np.loadtxt(filename)
    i, j, x, y, V = data.T
    NX = i.max()+1
    NY = j.max()+1
    shape = (int(NX), int(NY))
    i = i.reshape(shape)
    j = j.reshape(shape)
    x = x.reshape(shape)
    y = y.reshape(shape)
    V = V.reshape(shape)

    plt.title(filename)
    plt.contourf(x, y, V)
    plt.colorbar()
    plt.show()
