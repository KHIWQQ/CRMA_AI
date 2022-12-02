#Scipy
from scipy import constants
from scipy.optimize import root
from math import cos
from scipy.optimize import minimize
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import UnivariateSpline
from scipy import io
from scipy.stats import skew, kurtosis

#Insert the correct syntax for printing the kilometer unit (in meters).
print(constants.kilo)
#Insert the correct syntax for printing the kilometer unit (in meters).
print(constants.gram)
#Insert the correct syntax for printing the miles-per-hour unit (in meters per seconds).
print(constants.mph)
#constants.gram
#0.001
#constants.gram
#0.0254
#Insert the missing parts to print the square root of the equation:
def eqn(x):
  return x + cos(x)

myroot = root(eqn, 0)
print(myroot.x)

#Insert the missing parts to minimize the equation
def eqn(x):
  return x**2 + x + 2
mymin = minimize(eqn, 0, method='BFGS')

#Insert the missing parts to minimize the equation, by using the TNC method
def eqn(x):
  return x**2 + x + 2
mymin = minimize(eqn, 0, method='TNC')

#Insert the missing method to print the number of values in the array that are NOT zeros
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
print(csr_matrix(arr).count_nonzero())

#Insert the missing method to delete the zero values from the array
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
mat = csr_matrix(arr)
mat.eliminate_zeros()

print(mat)

#Insert the missing method to convert from csr (Compressed Sparse Row) to csc (Compressed Sparse Column)
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])

newarr = csr_matrix(arr).tocsc()
print(newarr)

#Insert the missing method to find all the connected components
arr = np.array([
  [0, 1, 2],
  [1, 0, 0],
  [2, 0, 0]
])
newarr = csr_matrix(arr)
print(connected_components(newarr))

#Insert the missing method to find the shortest path in a graph from one element to another
arr = np.array([
  [0, 1, 2],
  [1, 0, 0],
  [2, 0, 0]
])
newarr = csr_matrix(arr)
print(dijkstra(newarr, return_predecessors=True, indices=0))

#Which method is most likely to be used to find the smallest polygon that covers all of the given points?
#The ConvexHull() method
#Which method is most likely to be used to generate these triangulations through points?
#The Delaunay() method

#Insert the missing method to find the univariate spline interpolation
xs = np.arange(10)
ys = xs**2 + np.sin(xs) + 1
interp_func = UnivariateSpline(xs, ys)

#Insert the missing method to export data in Matlab format
arr = np.arange(10)
io.savemat('arr.mat', {"vec": arr})

#Insert the missing method to import data from a Matlab file
# Import:
mydata = io.loadmat('arr.mat')
print(mydata)

#Insert the missing method to meassure the summetry in data
v = np.random.normal(size=100)
print(skew(v))

#Insert the missing method to meassure whether the data is heavy or lightly tailed compared to a normal distribution
v = np.random.normal(size=100)
print(kurtosis(v))