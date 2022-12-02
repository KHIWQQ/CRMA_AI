from scipy import constants
print(constants.liter)
print(constants.pi)
print(dir(constants))

# Metric (SI) Prefixes:
print('Metric (SI) Prefixes:')
print(constants.yotta) #1e+24
print(constants.zetta) #1e+21
print(constants.exa) #1e+18
print(constants.peta) #1000000000000000.0
print(constants.tera) #1000000000000.0
print(constants.giga) #1000000000.0
print(constants.mega) #1000000.0
print(constants.kilo) #1000.0
print(constants.hecto) #100.0
print(constants.deka) #10.0
print(constants.deci) #0.1
print(constants.centi) #0.01
print(constants.milli) #0.001
print(constants.micro) #1e-06
print(constants.nano) #1e-09
print(constants.pico) #1e-12
print(constants.femto) #1e-15
print(constants.atto) #1e-18
print(constants.zepto) #1e-21
print()

# BinaryPrefixes:
print('BinaryPrefixes:')
print(constants.kibi) #1024
print(constants.mebi) #1048576
print(constants.gibi) #1073741824
print(constants.tebi) #1099511627776
print(constants.pebi) #1125899906842624
print(constants.exbi) #1152921504606846976
print(constants.zebi) #1180591620717411303424
print(constants.yobi) #1208925819614629174706176
print()

# Speed:
print('Speed:')
print(constants.kmh) #0.2777777777777778
print(constants.mph) #0.44703999999999994
print(constants.mach) #340.5
print(constants.speed_of_sound) #340.5
print(constants.knot) #0.5144444444444445
print()

# SciPy Optimizers
from scipy.optimize import root
from math import cos
def eqn(x):
    return x + cos(x)
myroot = root(eqn, 0)
print(myroot.x)
print()

# SciPy Spatial Data
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
points = np.array([
[2, 4],
[3, 4],
[3, 0],
[2, 2],
[4, 1]
])
simplices = Delaunay(points).simplices
plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color='r')
plt.show()
print()

# Statistical Description of Data
from scipy.stats import describe
v = np.random.normal(size=100)
res = describe(v)
print(res)