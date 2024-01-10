import numpy as np

coor = np.random.rand(20, 5)
value = np.random.rand(20)
print(coor)
print(value)
print(np.gradient(value, coor[:, 1]))