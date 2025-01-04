import numpy as np

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

def predict_softmax(A, w, b):
    """
    Gives probabilites of each class for given features and parameters.

    Args:
        A = Features (matrix of size(m, n))
        w = Parameters (matrix of size(n, num_classes))
        b = Parameters (array of size (num_classes, ))

    Returns:
        y_prob = Probabilities for each training example (matrix of size (m, num_classes))
    """
    y_ret = np.matmul(A, w) + b
    y_ret = softmax(y_ret)
    return y_ret

def softmax_to_class(y_prob, classes = None):
    """
    Gives most probable class for given probabilites of each class.

    Args:
        y_prob = Probabilities for each training example (matrix of size (m, num_classes))
        classes (optional) = If not [0,1,...,num_classes-1], the array of classes (array of size (num_classes, ))

    Returns:
        y_class = Predicted class for each training example (array of size (m, ))
    """
    y_ret = np.argmax(y_prob, axis = 1)
    try:
        if (classes == None):
            return y_ret
    except:
        y_ret = classes[y_ret]
        return y_ret

def predict_class(A, w, b, classes = None):
    """
    Gives most probable class for given features and parameters.

    Args:
        A = Features (matrix of size(m, n))
        w = Parameters (matrix of size(n, num_classes))
        b = Parameters (array of size (num_classes, ))
        classes (optional) = If not [0,1,...,num_classes-1], the array of classes (array of size (num_classes, ))

    Returns:
        y_class = Predicted class for each training example (array of size (m, ))
    """
    y_ret = predict_softmax(A, w, b)
    y_ret = softmax_to_class(y_ret, classes)
    return y_ret

def softmax_cost(y, y_prob, epsilon = 0):
    """
    Calculates Cross-Entropy cost for given features, parameters and targets.

    Args:
        y = Targets (array of size (m, ))
        y_prob = Probabilities for each training example (matrix of size (m, num_classes))
    
    Returns:
        cost (scalar)
    """
    m = np.shape(y)[0]
    cost = np.sum(-np.log(y_prob[np.arange(m),y] + epsilon)) / m
    return cost

def softmax_gradient_descent(A, w, b, y, n_iter, alpha, classes = None):
    """
    Performs softmax gradient descent.

    Args:
        A = Features (matrix of size(m, n))
        w = Parameters (matrix of size(n, num_classes))
        b = Parameters (array of size (num_classes, ))
        y = Targets (array of size (m, ))
        n_iter = Number of iterations (scalar)
        alpha = Learning rate (scalar)
        classes (optional) = If not [0,1,...,num_classes-1], the array of classes (array of size (num_classes, ))

    Returns:
        w_new = Parameters after gradient descent (matrix of size(n, num_classes))
        b_new = Parameters after gradient descent (array of size (num_classes, ))
        cost_history = Cost at each cycle
    """
    m, n = np.shape(A)
    num_classes = np.shape(w)[1]
    p10 = n_iter // 10
    p10c = 0
    cost_history = np.zeros(n_iter)
    A_trans = A.T
    y_onehot = np.zeros((m, num_classes))
    y_onehot[np.arange(m), y] = 1
    for i in range(n_iter):
        y_prob = predict_softmax(A, w, b)
        j = softmax_cost(y, y_prob)
        dj_db = np.sum(y_prob - y_onehot, axis = 0) / m
        dj_dw = np.matmul(A_trans, (y_prob - y_onehot)) / m
        b -= alpha * dj_db
        w -= alpha * dj_dw
        cost_history[i] = j
        if (i == p10c):
            print(f"Cost at iteration {i}: {j}")
            p10c += p10
    print(f"Cost at iteration {n_iter}: {j}")
    return w, b, cost_history

def mce(y, y_pred):
    """
    Calculates the multi-class classification error for a given model's predictions.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)

    Returns:
        MCE = Multi-class Classification Error (scalar)
    """
    m = np.shape(y)[0]
    err = np.sum((y == y_pred).astype(int)) / m
    return err

def one_f1_multi(y, y_pred, clas):
    """
    Calculates the multi-class F1 score for a class.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)
        clas = class whose F1 is to be computed

    Returns:
        F1_score (scalar)
    """
    tp = np.sum((y == y_pred) & (y == clas))
    fp = np.sum((y != y_pred) & (y_pred == clas))
    fn = np.sum((y != y_pred) & (y == clas))
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

def macro_f1_multi(y, y_pred):
    """
    Calculates the macro multi-class F1 score for a given model's predictions.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)

    Returns:
        F1_score (scalar)
    """
    classes = np.unique(y)
    f1 = 0
    for i in classes:
        f1 += one_f1_multi(y, y_pred, i)
    f1 /= np.shape(classes)[0]
    return f1

def metrics(y, y_pred, clas):
    """
    Calculates the true-positive,  false-positive and false-negative for a class.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)
        clas = class whose F1 is to be computed

    Returns:
        tp = Number of true-positive (scalar)
        fp = Number of false-positive (scalar)
        fn = Number of false-negative (scalar)
    """
    tp = np.sum((y == y_pred) & (y == clas))
    fp = np.sum((y != y_pred) & (y_pred == clas))
    fn = np.sum((y != y_pred) & (y == clas))
    return tp, fp, fn

def micro_f1_multi(y, y_pred):
    """
    Calculates the micro multi-class F1 score for a given model's predictions.

    Args:
        y = Targets (array-like)
        y_pred = Prediction (array-like)

    Returns:
        F1_score (scalar)
    """
    classes = np.unique(y)
    num_classes = np.shape(classes)[0]
    metri = np.zeros((num_classes,3))
    for i in range(num_classes):
        metri[i,:] = metrics(y, y_pred, classes[i])
    tp = np.sum(metri[:,0])
    fp = np.sum(metri[:,1])
    fn = np.sum(metri[:,2])
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1