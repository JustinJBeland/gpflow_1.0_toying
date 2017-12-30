import numpy as np
import math

def f(X, args=()):
	x = X[:, 0]
	y = X[:, 1]
	return np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + (5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10
	

