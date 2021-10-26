import numpy as np

def lotka(t, x, theta):
    y = np.empty(x.shape)
    y[0] =  x[0] * (theta[0] - theta[1] * x[1])
    y[1] = -x[1] * (theta[2] - theta[3] * x[0])
    return y


