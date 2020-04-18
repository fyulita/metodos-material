import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-4e-8, 4e-8, 1e-13)
y1 = (1 - np.cos(x)) / (x ** 2)
y2 = (2 * (np.sin(x / 2) ** 2)) / (x ** 2)

plt.title('Primera funcion')
plt.plot(x, y1)
plt.show()

plt.title('Segunda funcion')
plt.plot(x, y2)
plt.show()
