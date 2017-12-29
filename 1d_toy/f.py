import numpy as np
import math

def f(X, args=()):
	x = X[:, 0]
	return np.square(6.0 * x - 2.0) * np.sin(12.0 * x - 4.0)
	

