# -*- coding: utf-8 -*-
import math
from functools import reduce
from .utils import Vector
from .utils import sigmoid
from .linear_search import wolfe_powell
from .gradient_method import gradient_descent_framework


class LogisticRegression(object):
    def __init__(self, X, y):
        """Initialize dataset

        Args:
            X: list of tuple, (x1, x2, ..., xm)
            y: list, (y1, y2, ..., ym), yi should be 1 or -1
        """
        self.Xs = X
        self.ys = y
        self.w = None
        self.b = None

    def f(self, w):
        """Logistic Regression Loss function

        Args:
            w: tuple, coefficient and bias of logistic regression
        Returns:
            loss: float, loss of Logistic Regression
        """
        w, b = w[:-1], w[-1]
        # w take inner product with each x
        tmp = [Vector.inner(w, x) for x in self.Xs]
        tmp = Vector.add_constant(tmp, b)
        tmp = Vector.multiply_elemwise(self.ys, tmp)
        return -sum(math.log(sigmoid(x)) for x in tmp)

    def df(self, w):
        """Gradient of Logistic Regression Loss function

        Args:
            w: tuple, coefficient and bias of logistic regression
        Returns:
            grad: tuple, gradient of loss function at w
        """
        w, b = w[:-1], w[-1]
        tmp = [Vector.inner(w, x) for x in self.Xs]
        tmp = Vector.add_constant(tmp, b)
        tmp = Vector.multiply_elemwise(self.ys, tmp)
        tmp = list(sigmoid(x) for x in tmp)
        tmp = Vector.add_constant(tmp, -1.0)
        tmp = Vector.multiply_elemwise(tmp, self.ys)
        dw = [Vector.multiply(x, p) for x, p in zip(self.Xs, tmp)]
        dw = reduce(Vector.add, dw)
        db = sum(tmp)
        dw = dw + (db, )
        return dw
    
    def decline_direction(self, w):
        """Using Steepest Gradient descent algorithm

        Args:
            w: tuple, coefficient and bias of logistic regression
        Returns:
            d: tuple, decline direction at point w
        """
        return Vector.multiply(self.df(w), -1.0)

    def linear_search(self, f, df, xk, dk):
        """Using wolfe-powell algorithm to calculate forward step. 
        This function is called in gradient_descend_framework

        Args:
            f: callable, target function, self.f
            df: callable, gradient function at point (w, b), self.df
            xk: tuple, point at which to search decline step
            dk: tuple, direction at which to search decline step
        Returns:
            alpha_k: float, decline step size
        """
        return wolfe_powell(f, df, xk, dk, 1.0, 0.05, 0.05, 0.01, 0.5)

    def set(self, w, b):
        self.w = w
        self.b = b
        print("w={0}, b={1}, f(w,b)={2}".format(self.w, self.b, self.f(w+(b,))))
    
    def get(self):
        return self.w, self.b

    def fit(self, w0=None, eps=1e-6):
        """Fit Logistic Regression Model on self.Xs and self.ys

        Args:
            w0: tuple, initial point to start optimize
            eps: float, when norm of gradient is less than eps, the gradient descent algorithm will stop
        """
        if w0 is None:
            w0 = tuple(0 for _ in range(len(self.Xs[0])+1))
        wk = gradient_descent_framework(self.f, self.df, self.decline_direction, self.linear_search, w0, eps)
        self.set(wk[:-1], wk[-1])
    
    def predict(self, x):
        """Predict the probability that the label of sample x is 1

        Args:
            x: tuple, one sample
        Returns:
            prob: float, probability of the label of x is 1
        """
        return sigmoid(Vector.inner(self.w, x) + self.b)


if __name__ == "__main__":
    X = [
        (1, 2, 3, 4, 5, 6),
        (4, 5, 6, 7, 8, 9)
    ]
    y = [
        1,
        -1
    ]


    logistic_regression = LogisticRegression(X, y)
    logistic_regression.fit()
    for x in X:
        print(logistic_regression.predict(x))