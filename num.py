# Load in NumPy
import numpy as np

# The Basics
a = np.array([1,2,3], dtype='int32')
print(a)

b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)

# Get Dimension
print(a.ndim)

# Get Shape
print(b.shape)

# Get Type
print(a.dtype)

# Get Size (in bytes)
print(a.itemsize)

# Get total size (in bytes)
print(a.nbytes)

# Get number of elements
print(a.size)

# Accessing/Changing specific elements, rows, columns, etc
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)


a = np.array([1, 2, 3], dtype='int32')
print(a)

b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
print(b)

# Get Dimension
print(a.ndim)

# Get Shape
print(b.shape)

# Get Type
print(a.dtype)

# Get Size (in bytes)
print(a.itemsize)

# Get total size (in bytes)
print(a.nbytes)

# Get number of elements
print(a.size)

# Accessing/Changing specific elements, rows, columns, etc.
a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
print(a)

# Get a specific element [r, c]
print(a[1, 5])

# Get a specific row
print(a[0, :])

# Get a specific column
print(a[:, 2])

# Getting a little more fancy [startindex:endindex:stepsize]
print(a[0, 1:-1:2])

# Modifying elements
a[1, 5] = 20
a[:, 2] = [1, 2]
print(a)

# 3-D example
b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(b)


