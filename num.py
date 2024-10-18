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


# Load in NumPy
import numpy as np

# The Basics
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

# Get a specific element (work outside in)
print(b[0, 1, 1])

# Attempt to replace elements (incorrect size)
# This will throw a ValueError: setting an array element with a sequence
# Uncomment the next line to see the error:
# b[:, 1, :] = [[9,9,9], [8,8]]

# Correct replacement of elements
b[:, 1, :] = [[9, 9], [8, 8]]
print(b)

### Initializing Different Types of Arrays

# All 0s matrix
print(np.zeros((2, 3)))

# All 1s matrix
print(np.ones((4, 2, 2), dtype='int32'))

# Matrix filled with any other number
print(np.full((2, 2), 99))

# Matrix filled with a specific number, matching the shape of another array
print(np.full_like(a, 4))

# Random decimal numbers
print(np.random.rand(4, 2))

# Random integer values from -4 to 7 (size 3x3)
print(np.random.randint(-4, 8, size=(3, 3)))

# The identity matrix
print(np.identity(5))

# Repeat an array multiple times
arr = np.array([[1, 2, 3]])
r1 = np.repeat(arr, 3, axis=0)
print(r1)

# Create a 5x5 matrix filled with ones
output = np.ones((5, 5))
print(output)

# Create a 3x3 matrix filled with zeros and set the middle element to 9
z = np.zeros((3, 3))
z[1, 1] = 9
print(z)

# Replace the center of the 5x5 matrix with the 3x3 matrix
output[1:-1, 1:-1] = z
print(output)

##### Be careful when copying arrays!!!

# When you copy arrays, modifying one will not modify the other
a = np.array([1, 2, 3])
b = a.copy()
b[0] = 100
print(a)  # Original array remains unchanged

### Mathematics

# Basic mathematical operations on arrays
a = np.array([1, 2, 3, 4])
print(a)

# Add a scalar value to the array
print(a + 2)

# Subtract a scalar value
print(a - 2)

# Multiply by a scalar
print(a * 2)

# Divide by a scalar
print(a / 2)
# Import the numpy library for array operations
import numpy as np

# Array Addition
b = np.array([1, 0, 1, 0])  # Creating an array 'b'
a = np.array([1, 0, 3, 0])  # Creating an array 'a'
a + b  # Adding the arrays element-wise: array([1+1, 0+0, 3+1, 0+0]) => array([2, 0, 4, 0])

# Element-wise squaring of array 'a'
a = np.array([1, 2, 3, 4])  # Creating an array 'a'
a ** 2  # Squaring each element of 'a': array([1^2, 2^2, 3^2, 4^2]) => array([1, 4, 9, 16])

# Taking the cosine of elements in array 'a'
np.cos(a)  # Cosine of each element of 'a': np.cos([1, 2, 3, 4]) => array([cos(1), cos(2), cos(3), cos(4)])

# Matrix Multiplication
a = np.ones((2, 3))  # Creating a 2x3 matrix of ones
print(a)  # Display the matrix 'a': [[1. 1. 1.], [1. 1. 1.]]
b = np.full((3, 2), 2)  # Creating a 3x2 matrix filled with 2's
print(b)  # Display the matrix 'b': [[2 2], [2 2], [2 2]]
np.matmul(a, b)  # Matrix multiplication of 'a' and 'b' (2x3 * 3x2) => 2x2 result: [[6., 6.], [6., 6.]]

# Determinant of a matrix
c = np.identity(3)  # Creating a 3x3 identity matrix
np.linalg.det(c)  # Finding the determinant of the identity matrix 'c' => 1.0

# Statistics - Minimum, Maximum, and Sum
stats = np.array([[1, 2, 3], [4, 5, 6]])  # Creating a 2x3 matrix 'stats'
np.min(stats)  # Finding the minimum value in 'stats' => 1
np.max(stats, axis=1)  # Finding the maximum value along each row => array([3, 6])
np.sum(stats, axis=0)  # Summing elements column-wise => array([5, 7, 9])

# Reshaping an array
before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2x4 matrix 'before'
print(before)  # Display the original matrix
try:
    after = before.reshape((2, 3))  # Trying to reshape the 2x4 matrix to 2x3 (throws an error)
    print(after)
except ValueError as e:
    print(e)  # Handle the reshaping error as the shape is incompatible

# Stacking arrays vertically
v1 = np.array([1, 2, 3, 4])  # Creating vector 'v1'
v2 = np.array([5, 6, 7, 8])  # Creating vector 'v2'
np.vstack([v1, v2, v1, v2])  # Vertically stacking 'v1' and 'v2': creates a 4x4 matrix
# Output: [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]


import numpy as np

# Element-wise addition of arrays
a = np.array([1, 2, 3, 4])
b = np.array([1, 0, 1, 0])
print("Array addition:", a + b)  # Output: array([2, 2, 4, 4])

# Squaring the elements of array `a`
print("Element-wise square of a:", a ** 2)  # Output: array([1, 4, 9, 16], dtype=int32)

# Taking the cosine of each element in `a`
print("Cosine of a:", np.cos(a))  # Output: array([ 0.5403, -0.4161, -0.9900, -0.6536])

# Linear Algebra: Matrix multiplication using `np.matmul()`
a = np.ones((2, 3))
print("Matrix a (all ones):\n", a)

b = np.full((3, 2), 2)
print("Matrix b (all twos):\n", b)

result = np.matmul(a, b)
print("Matrix multiplication result:\n", result)  # Output: array([[6., 6.], [6., 6.]])

# Find the determinant of an identity matrix
c = np.identity(3)
print("Determinant of identity matrix:", np.linalg.det(c))  # Output: 1.0

# Statistics: Min, Max, and Sum operations on arrays
stats = np.array([[1, 2, 3], [4, 5, 6]])
print("Stats array:\n", stats)

print("Minimum value in stats:", np.min(stats))  # Output: 1
print("Maximum value along axis 1:", np.max(stats, axis=1))  # Output: array([3, 6])
print("Sum along axis 0:", np.sum(stats, axis=0))  # Output: array([5, 7, 9])

# Reorganizing Arrays: Reshape and Stack
before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("Before reshaping:\n", before)

# Attempt to reshape (will raise error due to size mismatch)
try:
    after = before.reshape((2, 3))
    print("After reshaping:\n", after)
except ValueError as e:
    print("Error:", e)

# Vertically stacking vectors using `vstack`
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])
vstack_result = np.vstack([v1, v2, v1, v2])
print("Vertically stacked arrays:\n", vstack_result)

# Horizontally stacking arrays using `hstack`
h1 = np.ones((2, 4))
h2 = np.zeros((2, 2))
hstack_result = np.hstack((h1, h2))
print("Horizontally stacked arrays:\n", hstack_result)

# Loading data from a file using `np.genfromtxt()`
# Assuming the file 'data.txt' exists in the same directory
filedata = np.genfromtxt('data.txt', delimiter=',').astype('int32')
print("Loaded data:\n", filedata)

# Boolean Masking and Advanced Indexing
boolean_mask = ~((filedata > 50) & (filedata < 100))
print("Boolean mask:\n", boolean_mask)



