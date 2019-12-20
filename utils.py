# -*- coding: utf-8 -*-
import math


class Vector(object):
    @staticmethod
    def add(v1, v2, fac1=1.0, fac2=1.0):
        """Calculate vector addition c=p*a+q*b

        Args:
            v1: tuple, vector a
            v2: tuple, vector b
            fac1: float, scale of v1
            fac2: float scale of v2
        Returns:
            c: tuple, c=fac1*v1+fac2*v2
        """
        return tuple(fac1*i+fac2*j for i, j in zip(v1, v2))

    @staticmethod
    def add_constant(vec, scal):
        """Add scal to each element of vec

        Args:
            vec: tuple, vector
            scal: float, scalar to be added
        Returns:
            out: tuple, result vector
        """
        return tuple(x+scal for x in vec)

    @staticmethod
    def inner(v1, v2):
        """Calculate vector inner product c=a^T*b
    
        Args:
            v1: tuple, vector a
            v2: tuple, vector b
        Returns:
            c: float, c=v1^T*v2
        """
        return sum(map(lambda x, y: x*y, v1, v2))

    @staticmethod
    def multiply(v, frac=1.0):
        """Calculate c=k*a
        Args:
            v: tuple, vector a
            frac: float, scalar k
        Returns:
            c: tuple, c=k*a
        """
        return tuple(x*frac for x in v)

    @staticmethod
    def multiply_elemwise(v1, v2, frac1=1.0, frac2=1.0):
        """Calculate c=p*a*q*b

        Args:
            v1: tuple, vector a
            v2: tuple, vector b
            frac1: float, p
            frac2: float, q
        Returns:
            c: tuple, result vector
        """
        return tuple(frac1*x*frac2*y for x, y in zip(v1, v2))

    @staticmethod
    def norm(v):
        """Calculate the norm of a vector
        
        Args:
            v: tuple, a vector
        Returns:
            norm: float, the norm of vector v
        """
        return sum(x*x for x in v) ** 0.5


def sigmoid(x):
    """Sigmoid function f(x)=1/(1+e^(-x))
    
    Args:
        x: float, input of sigmoid function
    Returns:
        y: float, function value
    """
    # for x which is to large, the sigmoid returns 1
    if x > 100:
        return 1.0

    # for x which is very small, the sigmoid returns 0
    if x < -100:
        return 1e-6
    
    return 1.0 / (1.0 + math.exp(-x))
    