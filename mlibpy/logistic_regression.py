import numpy as np
from mlibpy import linear_regression as lnr # type: ignore

def sigmoid(A):
    """
    Applies sigmoid function to the given matrix.

    Args:
        A = Prediction (matrix)

    Returns:
        A_sigmoid = Sigmoid of Prediction (matrix)
    """
    A = 1/(1 + np.exp(A*-1))
    return A

def decision(A, boundary = 0.5):
    """
    Returns value in binary when provided with probability.

    Args:
        A = Probability (matrix)
        boundary = Value where the decision boundary lies (scalar)

    Returns:
        A_result = Binary result (matrix)
    """
    B = (A >= boundary).astype(int)
    return B

def logistic_cost(y, y_pred, epsilon = 0):
    """
    Computes logistic cost.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)
    
    Returns:
        cost = Cost (scalar)
    """
    m = np.shape(y)[0]
    cost = np.sum(-(y*np.log(y_pred + epsilon) + (1-y)*np.log(1-y_pred + epsilon)))
    cost /= m
    return cost

def logistic_gradient_descent(A, w, b, y, n_iter, alpha):
    """
    Performs logistic gradient descent.

    Args:
        A = Features (matrix of shape (m, n))
        w = Parameters (array-like of shape (n,))
        b = Parameter (scalar)
        y = Targets (array-like of shape (m,))
        n_iter = Number of iterations (scalar)
        alpha = Learning rate (scalar)
    
    Returns:
        w_new = Parameters after gradient descent (array-like)
        b_new = Parameter after gradient descent (scalar)
        cost_history = Cost at each cycle
    """
    m = np.shape(A)[0]
    p10 = n_iter // 10
    p10c = 0
    cost_history = np.zeros(n_iter)
    for i in range(n_iter):
        y_pred = sigmoid(lnr.predict(A, w, b))
        j = logistic_cost(y, y_pred)
        diff = y_pred - y
        dj_dw = np.matmul(diff, A) / m
        dj_db = np.mean(diff)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost_history[i] = j
        if (i == p10c):
            print(f"Cost at iteration {i}: {j}")
            p10c += p10
    print(f"Cost at iteration {n_iter}: {j}")
    return w, b, cost_history

def bce(y, y_pred):
    """
    Calculates the binary classification error for a given model's predictions.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)

    Returns:
        BCE = Binary Classification Error (scalar)
    """
    m = np.shape(y)[0]
    yp = decision(y_pred)
    err = np.sum((yp == y).astype(int)) / m
    return err

def f1_score(y, y_pred):
    """
    Calculates the F1 score for a given model's predictions.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)

    Returns:
        F1_score (scalar)
    """
    m = np.shape(y)[0]
    yp = decision(y_pred)
    true_pos = np.sum((y == yp) & (y == 1))
    true_neg = np.sum((y == yp) & (y == 0))
    false_pos = np.sum((y != yp) & (y == 0))
    false_neg = np.sum((y != yp) & (y == 1))
    prec = true_pos / (true_pos + false_pos)
    rec = true_pos / (true_pos + false_neg)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1