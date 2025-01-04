import numpy as np

def normalize(A, Amean = None, Astdev = None, type = 0):
    """
    Normalizes data.

    Args:
        A = Features (matrix)
        type = 0 for z-score; 1 for mean
    
    Returns:
        An = Normalized features (matrix)
    """
    An = np.zeros_like(A)
    Amax = np.zeros_like(A[0])
    Amin = np.zeros_like(A[0])
    try:
        if Amean == None:
            Amean = np.zeros_like(A[0])
        if Astdev == None:
            Astdev = np.zeros_like(A[0])
    except:
        pass

    for i in range(np.shape(A)[1]):
        Amax[i] = np.max(A[:,i])
        Amin[i] = np.max(A[:,i])
        Amean[i] = np.mean(A[:,i])
        Astdev[i] = np.std(A[:,i], axis = 0)

    if (type == 0):
        An = (A - Amean) / Astdev
    else:
        An = (A - Amean) / (Amax - Amin)

    return An, Amean, Astdev

def predict(A, w, b):
    """
    Gives predictions for given features and parameters.

    Args:
        A = Features (matrix)
        w = Parameters (array-like)
        b = Parameter (scalar)

    Returns:
        yhat = Prediction (array-like)
    """

    yhat = np.matmul(A, w) + b
    return yhat

def find_cost(A, w, b, y):
    """
    Computes cost.

    Args:
        A = Features (matrix)
        w = Parameters (array-like)
        b = Parameter (scalar)
        y = Targets (array-like)
    
    Returns:
        cost = Cost (scalar)
    """
    m = np.shape(A)[0]
    loss = predict(A, w, b) - y
    cost = np.dot(loss, loss)
    cost /= (2 * m)
    return cost

def find_cost_regularized(A, w, b, y, lambdaa):
    """
    Computes cost with regularization.

    Args:
        A = Features (matrix)
        w = Parameters (array-like)
        b = Parameter (scalar)
        y = Targets (array-like)
        lambdaa = Coefficient of w_squared sum (scalar)
    
    Returns:
        cost = Cost (scalar)
    """
    m = np.shape(A)[0]
    loss = predict(A, w, b) - y
    cost = np.dot(loss, loss)
    cost += lambdaa * np.dot(w, w)
    cost /= (2 * m)
    return cost

def gradient_descent(A, w, b, y, n, alpha):
    """
    Performs gradient descent.

    Args:
        A = Features (matrix)
        w = Parameters (array-like)
        b = Parameter (scalar)
        y = Targets (array-like)
        n = Number of iterations (scalar)
        alpha = Learning rate (scalar)
    
    Returns:
        w_new = Parameters after gradient descent (array-like)
        b_new = Parameter after gradient descent (scalar)
        cost_history = Cost at each cycle
    """
    m = np.shape(A)[0]
    p10 = n // 10
    p10c = 0
    p_it = 0
    cost_history = np.zeros(n)
    for i in range(n):
        j = find_cost(A, w, b, y)
        diff = predict(A, w, b) - y
        dj_dw = np.matmul(diff, A) / m
        dj_db = np.sum(diff) / m
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost_history[i] = j
        if (i == p10c):
            print(f"Cost at iteration {i}: {j}")
            p10c += p10
            p_it += 1
    print(f"Cost at iteration {n}: {j}")
    return w, b, cost_history

def gradient_descent_regularized(A, w, b, y, n, alpha, lambdaa = 0):
    """
    Performs gradient descent with regularization.

    Args:
        A = Features (matrix)
        w = Parameters (array-like)
        b = Parameter (scalar)
        y = Targets (array-like)
        n = Number of iterations (scalar)
        alpha = Learning rate (scalar)
        lambdaa = Coefficient of w_squared sum (scalar)
    
    Returns:
        w_new = Parameters after gradient descent (array-like)
        b_new = Parameter after gradient descent (scalar)
        cost_history = Cost at each cycle
    """
    m = np.shape(A)[0]
    p10 = n // 10
    p10c = 0
    p_it = 0
    cost_history = np.zeros(n)
    for i in range(n):
        j = find_cost_regularized(A, w, b, y, lambdaa)
        diff = predict(A, w, b) - y
        dj_dw = np.matmul(diff, A) / m
        dj_db = np.sum(diff) / m
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost_history[i] = j
        if (i == p10c):
            print(f"Cost at iteration {i}: {j}")
            p10c += p10
            p_it += 1
    print(f"Cost at iteration {n}: {j}")
    return w, b, cost_history

def r2_score(y, y_pred):
    """
    Calculates the R-squared (R2) score for a given model's predictions.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)

    Returns:
        r2 = The R-squared score (scalar)
    """
    y_mean = np.mean(y)
    ssr = np.sum(np.square(y - y_pred))
    sst = np.sum(np.square(y - y_mean))
    r2 = 1 - (ssr/sst)
    return r2

def mse(y, y_pred):
    """
    Calculates the Mean Squared Error (MSE) for a given model's predictions.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)

    Returns:
        MSE = Mean squared error (scalar)
    """
    n = np.shape(y)[0]
    err = np.sum(np.square(y-y_pred)) / n
    return err