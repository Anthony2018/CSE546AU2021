import numpy as np
import pandas as pd

from utils import problem


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    d = data.shape[1]
    new_centers = np.zeros([num_centers, d])
    for num_center in range(num_centers):
        sub_index = [j for j in range(len(classifications)) if classifications[j] == num_center]
        sub_data = data[sub_index]
        center = np.mean(sub_data, axis=0)
        new_centers[num_center] = center
    return new_centers


@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    distance = dict()
    for i in range(len(centers)):
        distance[i] = np.square(np.linalg.norm(data - centers[i], axis=1))
    classifications = pd.DataFrame(distance).idxmin(axis=1)
    return classifications


@problem.tag("hw4-A")
def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """

    loss = 0.0
    distance = dict()
    for i in range(len(centers)):
        distance[i] = np.square(np.linalg.norm(data - centers[i], axis=1))
    distance_min = pd.DataFrame(distance).min(axis=1)
    loss = np.mean(np.sqrt(distance_min))
    return loss


@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> np.ndarray:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing trained centers.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """
    # intialized the data points
    n, d = data.shape
    k = num_centers
    first_center = np.random.choice(n, k, replace=False)

    center = data[first_center]
    center_new = center
    center_old = np.ones_like(center_new)
    error_list = []
    while (np.max(np.abs(center_new-center_old))) >= epsilon:
        center_old = center_new
        #find the new center
        classifications = cluster_data(data, center_old)
        center_new = calculate_centers(data, classifications, num_centers)

        #calcluate error
        error = calculate_error(data, center_new)
        print(error)
        error_list.append(error)
    return error_list, classifications, center_new

