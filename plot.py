import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.dat")
x, y, V = data.T
NX = x.max()+1
NY = y.max()+1
shape = (int(NX), int(NY))
x = x.reshape(shape)
y = y.reshape(shape)
V = V.reshape(shape)

plt.contourf(x, y, V)
plt.colorbar()
plt.show()
