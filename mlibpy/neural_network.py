import numpy as np
from mlibpy import linear_regression as lnr # type: ignore
from mlibpy import logistic_regression as logr # type: ignore
from mlibpy import softmax_regression as sftr # type: ignore

def relu(A):
    """
    Applies ReLU (Rectified Linear Unit) function to given input.

    Args:
        A = Input (Matrix, Array or scalar)
    
    Returns:
        A_relu = ReLU of Input (Same datatype as A)
    """
    return (A >= 0)*A

def diff_relu(A):
    """
    Calculates derivative of ReLU function.

    Args:
        A = Input (Matrix, Array or scalar)

    Returns:
        A_diff_relu = Derivative of ReLU function
    """
    return A >= 0

def sigmoid(A):
    """
    Applies sigmoid function to the given matrix.

    Args:
        A = Prediction (matrix)

    Returns:
        A_sigmoid = Sigmoid of Prediction (matrix)
    """
    An = 1/(1 + np.exp(A*-1))
    return An

def diff_sigmoid(A):
    """
    Calculates derivative of sigmoid function.

    Args:
        A = Input (Matrix, Array or scalar)

    Returns:
        A_diff_sig = Derivative of sigmoid function
    """
    temp = np.exp(-A)
    A_sig = temp / np.square(1 + temp)
    return A_sig

def linear(A):
    return A

def diff_linear(A):
    return np.ones_like(A)

def last_diff_linear(y, y_pred):
    return (y_pred - y)

def last_diff_sig(y, y_pred):
    return (y_pred - y)

def last_diff_sft(y, y_pred):
    m, num_classes = np.shape(y_pred)[0]
    y_onehot = np.zeros((m, num_classes))
    y_onehot[np.arange(m), y] = 1
    return y_pred - y_onehot

def softmax(yhat):
    """
    Returns probabilities after taking predictions as input

    Args:
        yhat = Predictions (matrix of size (m, num_classes))

    Returns:
        Prob = Probabilities for each training example (matrix of size (m, num_classes))
    """
    m = np.shape(yhat)[0]
    y_ret = np.exp(yhat)
    y_ret /= (np.sum(y_ret, axis = 1)).reshape(m,1)
    return y_ret


def initialize_parameters(A, n_layers, units):
    """
    Initializes parameters for the provided neural network architecture.

    Args:
        A = Dataset (matrix of size (m, n))
        n_layers = Number of layers in the neural network architecture
        units = Neurons in each layer (array of size (n_layers, ))

    Returns:
        w = List of parameters for each layer (list of length n_layers)
        b = List of parameters for each layer (list of length n_layers)
    """
    m, n = np.shape(A)
    w = list()
    b = list()
    w.append(np.random.rand(n, units[0]))
    b.append(np.zeros(units[0]))
    for i in range(n_layers - 1):
        w.append(np.random.uniform(-2, 2, (units[i], units[i+1])))
        b.append(np.zeros(units[i+1]))
    return w, b

def for_prop(A, w, b, activation):
    """
    Gives predictions for given features and parameters.

    Args:
        A = Dataset (matrix of size (m, n))
        w = List of parameters for each layer (list of length n_layers)
        b = List of parameters for each layer (list of length n_layers)
        activation = List of activation functions (list of length n_layers)
    
    Returns:
        T = Output of each layer without activation function (list of length n_layers)
        Z = Output of each layer with activation function (list of length n_layers)
    """
    m, n = np.shape(A)
    n_layers = len(b)
    T = list()
    Z = list()
    An = A.copy()
    for i in range(n_layers):
        An = np.matmul(An, w[i]) + b[i]
        T.append(An)
        An = activation[i](An)
        Z.append(An)
    return T, Z

cost = dict()
cost[linear] = lnr.find_cost
cost[sigmoid] = logr.logistic_cost
cost[softmax] = sftr.softmax_cost

diff = dict()
diff[linear] = diff_linear
diff[relu] = diff_relu
diff[sigmoid] = diff_sigmoid

last_diff = dict()
last_diff[linear] = last_diff_linear
last_diff[sigmoid] = last_diff_sig
last_diff[softmax] = last_diff_sft

def back_prop(A, w, b, y, activation, T, Z):
    """
    """
    m, n = np.shape(A)
    n_layers = len(activation)
    dw = list()
    db = list()
    if activation[-1] == softmax:
      dt = last_diff[activation[-1]](y, Z[-1])
    else:
      dt = last_diff[activation[-1]](y.reshape(m,1), Z[-1])
    db.append(np.sum(dt, axis=0) / m)
    dw.append(np.matmul(Z[-2].T, dt) / m)
    for i in range(1, n_layers - 1):
        dt = np.matmul(dt, w[-i].T) * diff[activation[-(i+1)]](T[-(i+1)])
        db.append(np.sum(dt, axis=0) / m)
        dw.append(np.matmul(Z[-(i+2)].T, dt) / m)
    dt = np.matmul(dt, w[-(n_layers - 1)].T) * diff[activation[-(n_layers)]](T[-(n_layers)])
    db.append(np.sum(dt, axis=0) / m)
    dw.append(np.matmul(A.T, dt) / m)
    dw.reverse()
    db.reverse()
    return dw, db


def nn_gradient_descent(A, w, b, y, activation, n_iter, alpha):
    """
    Performs gradient descent.

    Args:
        A = Dataset (matrix of size (m, n))
        w = List of parameters for each layer (list of length n_layers)
        b = List of parameters for each layer (list of length n_layers)
        y = Targets (array of size (m, ))
        activation = List of activation functions (list of length n_layers)
        n_iter = Number of iterations (scalar)
        alpha = Learning rate (scalar)
        
    Returns:
        w_new = List of parameters after gradient descent (list of length n_layers)
        b_new = List of parameters after gradient descent (list of length n_layers)
        cost_history = Cost at each cycle
    """
    m, n = np.shape(A)
    n_layers = len(b)
    p10 = n_iter // 10
    p10c = 0
    p_it = 0
    cost_history = np.zeros(n_iter)
    for i in range(n_iter):
        T, Z = for_prop(A, w, b, activation)
        if activation[-1] == linear:
            j = cost[linear](y, Z[-1].reshape(m,))
        elif activation[-1] == sigmoid:
            j = cost[sigmoid](y, Z[-1].reshape(m,), 1e-5)
        elif activation[-1] == softmax:
            j = cost[softmax](y, Z[-1], 1e-5)
        dw, db = back_prop(A,w,b,y,activation,T,Z)
        for k in range(n_layers):
            w[k] -= alpha*dw[k]
            b[k] -= alpha*db[k]
        cost_history[i] = j
        if (i == p10c):
            print(f"Cost at iteration {i}: {j}")
            p10c += p10
            p_it += 1
    print(f"Cost at iteration {n_iter}: {j}")
    return w, b, cost_history