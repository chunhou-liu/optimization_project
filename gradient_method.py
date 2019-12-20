# -*- coding: utf-8 -*-
from .utils import Vector


def gradient_descent_framework(
    f, 
    df,
    decline_direction, 
    linear_search, 
    x0, 
    eps
):
    """Gradient descent algorithm framework

    Args:
        f: callable, target function, take x0 as input and output a scalar
        df: callable, gradient of target function, take x0 as input and 
            returns a tuple as the gradient of f at x0
        decline_direction: callable, algorithm determining decreasing direction of f at point xk, 
            take xk as input and returns a tuple as the decreasing direction at point xk
        linear_search: callable, linear search algorithm, take f, df, xk, dk as input and
            returns the a scalar as the step to go at point xk and direction dk
        x0: tuple, initial point to start
        eps: float, positive value, when norm(df(xk))<=eps, the algorithm will stop search
    Returns:
        x_star: tuple, points at which f takes the local minimum with precision of eps
    Raises:
        ValueError
    """
    # arguments check
    if eps <= 0:
        raise ValueError("eps should be positive, got {0} instead".format(eps))

    # program framework
    xk = x0
    while Vector.norm(df(xk)) > eps:
        dk = decline_direction(xk)
        alpha_k = linear_search(f, df, xk, dk)
        # print("xk:", xk, "f(xk):", f(xk), "dk:", dk, "alpha_k:", alpha_k)
        print("f(xk):", f(xk))
        xk = Vector.add(xk, dk, 1.0, alpha_k)
    return xk


if __name__ == "__main__":
    def f(x):
        return (x[0]-2)**4+(x[1]-1)**4+(x[2]-1)**2
    
    def df(x):
        return (4*(x[0]-2)**3, 4*(x[1]-1)**3, 2*(x[2]-1))
    
    def decline_direction(x):
        return Vector.multiply(df(x), -1.0)

    def decline_direction_newton(x):
        # Newton algorithm
        return ((2-x[0])/3.0, (1-x[1])/3.0, 1-x[2])
    
    def linear_search_armijo(f, df, xk, dk):
        from .linear_search import armijo
        return armijo(f, df, xk, dk, 1.0, 0.3, 0.5)
    
    def linear_search_wolfe_powell(f, df, xk, dk):
        from .linear_search import wolfe_powell
        return wolfe_powell(f, df, xk, dk, 1.0, 0.3, 0.2, 0.4, 0.5)

    x0 = (500, -100, -2)
    eps = 1e-8
    x_star_armijo = gradient_descent_framework(f, df, decline_direction_newton, linear_search_armijo, x0, eps)
    x_star_wolfe_powell = gradient_descent_framework(f, df, decline_direction_newton, linear_search_wolfe_powell, x0, eps)
    print("armijo:", x_star_armijo, "wolfe-powell:", x_star_wolfe_powell)