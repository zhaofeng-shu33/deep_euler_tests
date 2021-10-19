import numpy as np
def lotka(t, x):
    y = np.empty(x.shape)
    y[0] =  x[0] - x[0]*x[1]
    y[1] = -x[1] + x[0]*x[1]
    return y
