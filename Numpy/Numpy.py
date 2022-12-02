#Create a NumPy ndarray Object
import numpy
arr = numpy.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

#Dimension in Arrays
#0-D Arrays
import numpy as np
arr_0d = np.array(42)
print(arr_0d)
#1-D Arrays
import numpy as np
arr_1d = np.array([1, 2, 3, 4, 5])
print(arr_1d )
#2-D Arrays
import numpy as np
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d)
#3-D Arrays
import numpy as np
arr_3d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr_3d)

#check Number of Dimension
import numpy as np
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

#Numpy Array Indexing
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[0])
print(arr[1])
print(arr[2] + arr[3])
arr2 = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('2nd element on 1st dim: ', arr2[0, 1])
print('5th element on 2nd dim: ', arr2[1, 4])
arr3 = np.array([[[1, 2, 3], [4, 5, 6]],
[[7, 8, 9], [10, 11, 12]]])
print(arr3[0, 1, 2])

#Negative Indexing
import numpy as np
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('Last element from 2nd dim: ', arr[1, -1])

#Numpy Array Slicing
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5]) #->1
print(arr[4:]) #->2
print(arr[-3:-1]) #->3
print(arr[1:5:2]) #->4
print(arr[::2]) #->5

#Slicing 2-D Arrays
import numpy as np
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[1, 1:4]) #->1
print(arr[0:2, 2]) #->2
print(arr[0:2, 1:4]) #->3

#Numpy Data types
import numpy as np
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array(['apple', 'banana', 'cherry'])
print(arr1.dtype)
print(arr2.dtype)
arr3 = np.array([1, 2, 3, 4], dtype='S')
print(arr3)
print(arr3.dtype)

#NumPy Array Copy vs View
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)
x2 = arr.view()
arr[0] = 22
print(x2)

#NumPy Array Shape
import numpy as np
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr.shape)

#NumPy Array Reshaping
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
newarr2 = arr.reshape(2, 3, 2)
print(newarr)
print(newarr2)

#NumPy Joining Array
import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)

#Random Permutations of Elements
from numpy import random
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
random.shuffle(arr)
print(arr)

from numpy import random
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(random.permutation(arr))

#Normal (Gaussian) Distribution
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
x = random.normal(size=(2, 3))
print(x)
x2 = random.normal(loc=1, scale=2, size=(2, 3))
print(x2)
sns.distplot(random.normal(size=1000),hist=False)
plt.show()