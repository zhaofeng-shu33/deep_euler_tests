import numpy as np

def lotka(t, x, theta):
    y = np.empty(x.shape)
    y[0] =  x[0] * (theta[0] - theta[1] * x[1])
    y[1] = -x[1] * (theta[2] - theta[3] * x[0])
    return y


def lotka_old(t, x):
    y = np.empty(x.shape)
    y[0] =  x[0] - x[0]*x[1]
    y[1] = -x[1] + x[0]*x[1]
    return y

def lotka_v2(t, x):
    y = np.empty(x.shape)
    y[0] =  x[0] * (1.5 - x[1])
    y[1] = -x[1] * (3 - x[0])
    return y

if __name__ == '__main__':
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    sol = solve_ivp(lotka_old, [0, 10], [2.0, 1.0], rtol=1e-6, atol=1e-9)
    plt.plot(sol.y[0,:], sol.y[1, :])
    #sol = solve_ivp(lotka_v2, [0, 10], [5, 5], rtol=1e-6, atol=1e-9)
    #plt.plot(sol.y[0,:], sol.y[1, :])
    plt.show()
