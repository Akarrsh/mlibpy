import numpy as np
from itertools import combinations_with_replacement
from mlibpy import linear_regression as lnr # type: ignore

def gen_poly(X, degree):
    """
    Generate polynomial features up to a given degree.

    Args:
    X = Features (matrix)
    degree = Maximum degree of polynomial terms (scalar)

    Returns:
    polynomial = matrix
    """
    m, n = np.shape(X)
    temp_array = np.zeros(m)

    cnt = 0
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n), d):
            cnt+=1
    polynom = np.zeros((m,cnt))

    cnt = 0
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n), d):
            polynom[:,cnt] = np.prod(X[:, comb], axis=1)
            cnt+=1
    
    return polynom

def poly_regression(X, y, degree, n_iter, alpha, lambdaa, X_mean = None, X_stdev = None):
    """
    Takes features, targets and other paramaters and performs gradient descent.

    Args:
    X = Features (matrix)
    y = Targets (array-like)
    degree = Maximum degree of polynomial terms (scalar)
    n_iter = Number of iterations (scalar)
    alpha = Learning rate (scalar)
    lambdaa = Coefficient of w_squared sum (scalar)

    Returns:
    Xn_poly = Normalised polynomial features (matrix)
    n = Number of features (scalar)
    w = Parameters after gradient descent (array-like)
    b = Parameter after gradient descent (scalar)
    cost_history = Cost at 10% intervals
    yhat = Prediction (array-like)
    """
    X = gen_poly(X, degree)
    X, X_mean, X_stdev = lnr.normalize(X, X_mean, X_stdev)
    m,n = np.shape(X)
    w = np.zeros(n)
    b = 0
    (w, b, cost_history) = lnr.gradient_descent_regularized(X, w, b, y, n_iter, alpha, lambdaa)
    yhat = lnr.predict(X, w, b)
    return X, w, b, cost_history, yhat, X_mean, X_stdev