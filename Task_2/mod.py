import numpy as np

def coord_tx():
    size = 20
    a = -3
    b = 4
    c = 1
    t = np.random.random(size)
    t.sort()
    x = a*t**2 + b*t + c
    return t, x

def mse(a,b):
    return ((a - b)**2).sum()/len(a)