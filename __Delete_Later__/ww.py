from ANNPDE.shapes import ElipseShape
import numpy as np
from random import randint
import numpy as np
import numpy as np

a = ElipseShape(randint(1, 200), 200, 5000, 10, [0, 0], 1, 2)
a.get()
# a.plot(np.ones(200), np.ones(5000))


# Generate a 5D array of random numbers
random_data = np.random.rand(200, 5)

x = np.linspace(0, 2*np.pi, 3)
y = np.linspace(0, 2*np.pi, 2)
t = np.linspace(0, 1, 10)
print(x)
print(y)
print(t)

X, Y, T = np.meshgrid(x, y, t, indexing='ij')
Z = np.sin(X) * np.cos(Y) * np.exp(T)
print(X)
print(Y.shape)
print(T.shape)
print(Z.shape)


grad = np.gradient(random_data)
# Calculate the derivative of the third value of u on the second column of x
u = random_data[:, 2]
x_column = random_data[:, 1]
derivative = np.gradient(u, x_column)
print(derivative)


