# -*- coding: utf-8 -*-

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
            frac: floar, scalar k
        Returns:
            c: tuple, c=k*a
        """
        return tuple(x*frac for x in v)

    @staticmethod
    def norm(v):
        """Calculate the norm of a vector
        
        Args:
            v: tuple, a vector
        Returns:
            norm: float, the norm of vector v
        """
        return sum(x*x for x in v) ** 0.5