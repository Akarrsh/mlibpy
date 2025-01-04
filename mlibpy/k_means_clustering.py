import numpy as np
import matplotlib.pyplot as plt

colors = ["Red", "Green", "Blue", "Pink", "Orange",
    "Purple", "Lime", "Fuchsia", "Aqua", "Teal",
    "Coral", "Gold", "Turquoise", "Salmon", "Violet",
    "Indigo", "DarkRed", "DarkGreen", "DarkBlue", "MediumPurple",
    "Crimson", "SlateBlue", "RoyalBlue", "MediumOrchid", "Olive",
    "YellowGreen", "DeepPink", "CadetBlue", "OrangeRed", "MediumSlateBlue",
    "LightBlue", "HotPink", "MediumSeaGreen", "SteelBlue", "DarkOrange",
    "Chocolate", "FireBrick", "SlateGray", "DarkViolet", "MediumTurquoise",
    "SpringGreen", "Lavender", "PeachPuff", "Goldenrod", "DarkKhaki",
    "LightCoral", "DodgerBlue", "Moccasin", "DarkOliveGreen", "LightSeaGreen"]

def initialize_centroids(A, num_clusters):
    """
    Initializes cluster centroids on the given dataset.

    Args:
        A = Dataset (matrix of size (m, n))
        num_clusters = Number of clusters (scalar)

    Returns:
        centroids = Coordinates of cluster centroids (matrix of size (num_clusters, n))
    """
    m, n = np.shape(A)
    indexes = np.random.choice(m, num_clusters, replace = False)
    centroids = A[indexes, :]
    return centroids

def assign_clusters(A, centroids):
    """
    Assigns clusters to all points in a dataset.

    Args:
        A = Dataset (matrix of size (m, n))
        centroids = Coordinates of cluster centroids (matrix of size (num_clusters, n))

    Returns:
        clusters = Index of assigned cluster per point of dataset (array of size (m, ))
    """
    m, n = np.shape(A)
    num_clusters = np.shape(centroids)
    dist = A[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist = np.sum(np.square(dist), axis = 2)
    clusters = np.argmin(dist, axis = 1)
    return clusters

def move_centroids(A, clusters, num_clusters):
    """
    Move the centroids to center of the cluster.

    Args:
        A = Dataset (matrix of size (m, n))
        clusters = Index of assigned cluster per point of dataset (array of size (m, ))
        num_clusters = Number of clusters (scalar)

    Returns:
        centroids = New coordinates of cluster centroids (matrix of size (num_clusters, n))
    """
    m, n = np.shape(A)
    centroids = np.zeros((num_clusters, n))
    for i in range(num_clusters):
        cluster_i = A[clusters == i]
        centroids[i] = np.mean(cluster_i, axis = 0)
        if np.isnan(np.mean(cluster_i, axis = 0)[0]):
            centroids[i] = A[np.random.randint(0, m), :]
    return centroids

def fit_clusters(A, num_clusters, n_iters_max):
    """
    Fit clusters on a datset.

    Args:
        A = Dataset (matrix of size (m, n))
        num_clusters = Number of clusters (scalar)
        n_iters_max = Maximum number of cycles of moving centroids and reassigning clusters (scalar)

    Returns:
        clusters = Index of assigned cluster per point of dataset (array of size (m, ))
        centroids = Coordinates of cluster centroids (matrix of size (num_clusters, n))
    """
    m, n = np.shape(A)
    centroids = initialize_centroids(A, num_clusters)
    centroids_prev = np.copy(centroids)
    for i in range(n_iters_max):
        clusters = assign_clusters(A, centroids)
        centroids = move_centroids(A, clusters, num_clusters)
        if np.array_equal(centroids, centroids_prev):
            break
        centroids_prev = np.copy(centroids)
    clusters = assign_clusters(A, centroids)
    return clusters, centroids

def plot_clusters(A, clusters, centroids, colors, feature_x, feature_y):
    """
    Plots points in dataset representing their clusters.

    Args:
        A = Dataset (matrix of size (m, n))
        clusters = Index of assigned cluster per point of dataset (array of size (m, ))
        centroids = Coordinates of cluster centroids (matrix of size (num_clusters, n))
        colors = Array of colors (array-like)
        feature_x = Feature to be plotted on x-axis (scalar in range [1, n])
        feature_y = Feature to be plotted on y-axis (scalar in range [1, n])
    """
    num_clusters = np.shape(centroids)[0]
    plt.figure(figsize = (10,10), dpi = 100)

    plt.title(f"Feature {feature_y} v/s Feature {feature_x}", fontdict = {'fontsize' : 20})
    plt.xlabel(f"Feature {feature_x}")
    plt.ylabel(f"Feature {feature_y}")

    fx = np.array(A[:, feature_x-1])
    fy = np.array(A[:, feature_y-1])

    for i in range(num_clusters):
        plot_fx = fx[clusters == i]
        plot_fy = fy[clusters == i]
        plt.scatter(plot_fx, plot_fy, color = colors[i])

    for i in range(num_clusters):
        plt.plot(centroids[i,feature_x-1], centroids[i,feature_y-1], color = colors[i], marker = 'X', markeredgecolor = 'black', markersize = 12)

    plt.show()

def cost_k_means(A, centroids, clusters):
    """
    Computes cost for the given cluster distribution.

    Args:
        A = Dataset (matrix of size (m, n))
        centroids = Coordinates of cluster centroids (matrix of size (num_clusters, n))
        clusters = Index of assigned cluster per point of dataset (array of size (m, ))

    Returns:
        cost = Cost for K-means clustering (scalar)
    """
    m, n = np.shape(A)
    num_clusters = np.shape(centroids)[0]
    cost = np.sum(np.square( A[np.arange(m), :] - centroids[clusters[np.arange(m)]] )) / m
    return cost

def random_initialization(A, num_clusters, n_sets, n_iters_per_set):
    """
    Fits many sets of clusters and returns the one with lowest cost.

    Args:
        A = Dataset (matrix of size (m, n))
        num_clusters = Number of clusters (scalar)
        n_sets = Number of sets of clusters computed (scalar)
        n_iters_per_set = Maximum number of cycles of moving centroids and reassigning clusters (scalar)

    Returns:
        clusters = Index of assigned cluster per point of dataset (array of size (m, ))
        centroids = Coordinates of cluster centroids (matrix of size (num_clusters, n))
    """
    m, n = np.shape(A)
    fin_clusters, fin_centroids = fit_clusters(A, num_clusters, n_iters_per_set)
    min_cost = cost_k_means(A, fin_centroids, fin_clusters)
    for i in range(n_sets - 1):
        clusters, centroids = fit_clusters(A, num_clusters, n_iters_per_set)
        cost = cost_k_means(A, centroids, clusters)
        if cost < min_cost:
            min_cost = cost
            fin_centroids = np.copy(centroids)
            fin_clusters = np.copy(clusters)
    return fin_clusters, fin_centroids