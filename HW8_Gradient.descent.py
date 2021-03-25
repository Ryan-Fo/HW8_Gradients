import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def func(x,y):
    """
    Returns the original function for this problem
    Input: x and y values
    Output: value of the function with respect to both values
    """
    return 5*np.exp(-1*(x - 1)**2 - 2*(y-3)**2)+3*np.exp(-2*(x-4)**2-1*(y-1)**2)

def partial_x(x,y):
    """
    Returns the analytical partial derivative of the function with respect to x
    Input: x and y
    Output: list of partial derivatives
    """
    return -12*(x-4)*np.exp(-2*(x-4)**2 - (y-1)**2)-10*(x-1)*np.exp(-1*(x-1)**2-2*(y-3)**2)
    

def partial_y(x,y):
    """
    Returns the analytical partial derivative of the function with respect to y
    Input: x and y 
    Output: list of partial derivatives
    """
    return -20*(y-3)*np.exp(-1*(x-1)**2-2*(y-3)**2)-6*(y-1)*np.exp(-2*(x-4)**2-(y-1)**2)


def grad_descent(func = None, deriv = None, start = None, gamma = None, max_iterations = None):
    count = 0
    x1, x2 = func, deriv
    for count in range(max_iterations):
        start = start - gamma * x2(start)
        i += 1
    return start

if __name__ == "__main__":

    xaxe = np.arange(0,6,0.1)
    yaxe = np.arange(0,6,0.1)
    xs,ys = np.meshgrid(xaxe,yaxe)
    gradient = func(xs,ys) #Z component for graph
    
    print(gradient,'\n')
    
    df_dx = partial_x(xaxe,yaxe)
    df_dy = partial_y(xaxe,yaxe)
    print('Partial W.R.T x:',df_dx,'\n')
    print('Partial W.R.T y:',df_dy,'\n')
    
    #None of this code worked. I know there were errors with the grad descent function.
    #The fun for lambda does not take two position arguements
    """
    fun = lambda x,y: 5*np.exp(-1*(x[0] - 1)**2 - 2*(y[0]-3)**2)+3*np.exp(-2*(x[0]-4)**2-1*(y[0]-1)**2)
    minima = minimize(fun,x0 = (2,3))
    print(minima.fun)
    
    
    using_descent_x = grad_descent(func = func(xs,ys), deriv = df_dx, start = 1, gamma = 0.1, max_iterations = 100)
    using_descent_y = grad_descent(func = func(xs,ys), deriv = df_dy, start = 1, gamma = 0.1, max_iterations = 100)
    
    print('For Gradient Descent WRT x:',using_descent_x,'\n')
    print('For Gradient Descent WRT y:',using_descent_y)
    """
    
    #Code for contour plot
    plt.contourf(xs,ys,gradient)
    plt.colorbar()
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title('F(x,y) = 5exp[-(x-1)^2 - 2(y-3)^2]+3exp[-2(x-4)^2 - (y-1)^2]')
    plt.show()