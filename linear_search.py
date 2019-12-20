# -*- coding: utf-8 -*-
from .utils import Vector

# This module implements linear search algorithms, include
#   1. armijo linear search
#   2. wolfe-powell linear search

def armijo(
    f,
    df,
    xk,
    dk,
    beta,
    rho,
    sigma_1
):
    '''Armijo Linear Search Algorithm.

    Args:
        f: callable, target function to be optimized, takes xk as input and returns a scalar
        df: callable, gradient function of target function, take xk as input and returns as tuple which has the same length as xk
        xk: tuple, point which the algorithm search at
        dk: tuple, decreasing direction of f at xk, must have the same length as xk
        beta: float greater than 0, initial search step
        rho: float in range(0, 1), fraction the search step declines at when find first alpha_k satisfying first constrain
        sigma_1: float in range (0, 1), parameter in first constraint
    Returns:
        alpha_k: float, step for xk to decline at direction d_k
    Raises:
        ValueError
    '''
        # argument check
    if len(dk) != len(xk):
        raise ValueError("length of d and xk should be equal, got len(d)={0}, len(xk)={1} instead.".format(len(dk), len(xk)))
    if beta <= 0.0:
        raise ValueError("beta should be positive, got {0} instead.".format(beta))
    if rho >= 1 or rho <= 0:
        raise ValueError("rho should be in range (0,1), got {0} instead.".format(rho))
    if sigma_1 >= 1 or sigma_1 <= 0:
        raise ValueError("sigma_1 should be in range (0, 1), got {0} instead.".format(sigma_1))
    # check if dk is decreasing direction
    if Vector.inner(df(xk), dk) >= 0:
        raise ValueError("dk should be decreasing direction")

    condition = lambda alpha: f(Vector.add(xk, dk, 1.0, alpha)) <= f(xk) + sigma_1*alpha*Vector.inner(df(xk), dk)
    # check for alpha_k = 1.0
    if condition(1.0):
        return 1.0
    alpha_k = beta
    while not condition(alpha_k):
        alpha_k *= rho
    return alpha_k


def wolfe_powell(
    f, 
    df, 
    xk,
    dk, 
    beta, 
    rho, 
    rho_1, 
    sigma_1, 
    sigma_2
):
    '''Wolfe-Powell Linear Search Algorithm.

    Args:
        f: callable, target function to be optimized, takes xk as input and returns a scalar
        df: callable, gradient function of target function, take xk as input and returns as tuple which has the same length as xk
        xk: tuple, point which the algorithm search at
        dk: tuple, decreasing direction of f at xk, must have the same length as xk
        beta: float greater than 0, initial search step
        rho: float in range(0, 1), fraction the search step declines at when find first alpha_k satisfying first constraint
        rho_1: float in range (0, 1), fraction the search step declines at when find the alpha_k satisfying second constraint
        sigma_1: float in range (0, 0.5), parameter in first constraint
        sigma_2: float in range(sigma_1, 1), parameter in first constraint
    Returns:
        alpha_k: float, step for xk to decline at direction d_k
    Raises:
        ValueError
    '''

    # argument check
    if len(dk) != len(xk):
        raise ValueError("length of d and xk should be equal, got len(d)={0}, len(xk)={1} instead.".format(len(dk), len(xk)))
    if beta <= 0.0:
        raise ValueError("beta should be positive, got {0} instead.".format(beta))
    if rho >= 1 or rho <= 0:
        raise ValueError("rho should be in range (0,1), got {0} instead.".format(rho))
    if rho_1 >= 1 or rho_1 <= 0:
        raise ValueError("rho_1 should be in range (0, 1), got {0} instead.".format(rho))
    if sigma_1 >= 0.5 or sigma_1 <= 0:
        raise ValueError("sigma_1 should be in range (0, 0.5), got {0} instead.".format(sigma_1))
    if sigma_2 <= sigma_1 or sigma_2 >= 1:
        raise ValueError("sigma_2 should be in range (sigma_1, 1), got {0} instead.".format(sigma_2))
    # check if dk is decreasing direction
    if Vector.inner(df(xk), dk) >= 0:
        raise ValueError("dk should be decreasing direction")
    
    condition_1 = lambda alpha: f(Vector.add(xk, dk, 1.0, alpha)) <= f(xk) + sigma_1 * alpha * Vector.inner(df(xk),dk)
    condition_2 = lambda alpha: Vector.inner(df(Vector.add(xk, dk, 1.0, alpha)), dk) >= sigma_2 * Vector.inner(df(xk), dk)
    
    # check for alpha_k = 1
    if condition_1(1.0) and condition_2(1.0):
        return 1.0
    
    # following two while loops find largest alpha_k which meets the first inequality
    # note:
    #   1. if beta meets condition_1 then first loop will execute 
    #      until alpha_k not meets condition_1 and the second will only execute once
    #   2. if beta doesn't meets condition_1 then first loop will not execute
    #      and second loop will execute until alpha_k meets condition_1
    alpha_k = beta
    while condition_1(alpha_k):
        alpha_k /= rho
    while not condition_1(alpha_k):
        alpha_k *= rho
    beta_k = alpha_k / rho
    
    # search until second condition is satisfied
    while not condition_2(alpha_k):
        alpha_k_prev, alpha_k, frac, delta_step = alpha_k, beta_k, rho_1, beta_k - alpha_k
        while not condition_1(alpha_k):
            alpha_k = alpha_k_prev + frac*delta_step
            frac *= rho_1
        beta_k = alpha_k / rho_1
    return alpha_k



if __name__ == "__main__":
    def f(x):
        return x[0]**2 + x[1]**2
    
    def df(x):
        return (2*x[0], 2*x[1])
    
    xk = (1.0, 1.0)
    dk = Vector.multiply(df(xk), -1.0)
    beta = 0.9
    rho, rho_1 = 0.1, 0.5
    sigma_1, sigma_2 = 0.3, 0.7
    alpha_k = wolfe_powell(f, df, xk, dk, beta, rho, rho_1, sigma_1, sigma_2)
    print("Wolfe-Powell alpha_k:", alpha_k)
    alpha_k = armijo(f, df, xk, dk, beta, rho, sigma_1)
    print("Armijo alpha_k:", alpha_k)