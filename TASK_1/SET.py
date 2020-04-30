import numpy as np

def coordinates_xy(): 
    
    """This function creates a set of points with size = size, 
    from the inteval [x_left, x_right) that obey the law y = a*x**2+b*x+c"""

    size = 10
    x_left, x_right  = [-5, 5]
    
    x = np.random.random_sample(size=size)*(x_right-x_left) + x_left 
    x.sort()
    
    a, b, c = [1, 2, 3]
    y = a*x**2+b*x+c       # the law 
    
    return x, y