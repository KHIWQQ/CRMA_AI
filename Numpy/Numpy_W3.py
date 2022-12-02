import numpy as np

# Numpy Creating Arrays
arr = np.array([1, 2, 3, 4, 5])
print(arr)
arr = np.array([1, 2, 3, 4], ndmin=2)
print(arr)
arr = np.array([1, 2, 3, 4])
print(arr.ndim)

# Numpy Creating Arrays
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])
arr = np.array([10, 20, 30, 40, 50])
print(arr[4])
arr = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
print(arr[1, 0])
arr = np.array([10, 20, 30, 40, 50])
print(arr[-1])

# Numpy Slicing Arrays
arr = np.array([10, 15, 20, 25, 30, 35, 40])
print(arr[1:4])
arr = np.array([10, 15, 20, 25, 30, 35, 40])
print(arr[2:4])
arr = np.array([10, 15, 20, 25, 30, 35, 40])
print(arr[1:4:2])
arr = np.array([10, 15, 20, 25, 30, 35, 40])
print(arr[::2])

# Numpy Data Types
"""i = integer 
b = boolean 
u = unsigned integer 
f = float 
c = complex float 
m = timedelta 
M = datetime 
O = object 
S = string"""
arr = np.array([1, 2, 3, 4])
print(arr.dtype)
arr = np.array([1, 2, 3, 4], dtype='S')
print(arr)
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype('i')
print(newarr)

# Numpy Copy VS View
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()

# Numpy Array Shape
arr = np.array([1, 2, 3, 4, 5])
print(arr.shape)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
print(newarr)
arr = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
newarr = arr.reshape(-1)
print(newarr)

# Numpy Array Join
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)

# Numpy Array Search
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x)

# Numpy Array Sort
arr = np.array([3, 2, 0, 1])
x = np.sort(arr)
print(x)