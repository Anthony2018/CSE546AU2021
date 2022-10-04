import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset, problem


@problem.tag("hw2-A")
def train(x: np.ndarray, y: np.ndarray, _lambda: float) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), targets (`y`) and regularization parameter (`_lambda`)
    to train a weight matrix $$\\hat{W}$$.


    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        y (np.ndarray): targets represented as `(n, k)` matrix.
            n is number of observations, k is number of classes.
        _lambda (float): parameter for ridge regularization.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: weight matrix of shape `(d, k)`
            which minimizes Regularized Squared Error on `x` and `y` with hyperparameter `_lambda`.
    """
    return np.linalg.solve(np.dot(x.T,x) + _lambda * np.eye(x.shape[1]), np.dot(x.T,y))


@problem.tag("hw2-A")
def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), and weight matrix (`w`) to generate predicated class for each observation in x.

    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        w (np.ndarray): weights represented as `(d, k)` matrix.
            d is number of features, k is number of classes.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: predictions matrix of shape `(n,)` or `(n, 1)`.
    """
    return np.argmax(np.dot(w.T, x.T), axis = 0)


@problem.tag("hw2-A")
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encode a vector `y`.
    One hot encoding takes an array of integers and coverts them into binary format.
    Each number i is converted into a vector of zeros (of size num_classes), with exception of i^th element which is 1.

    Args:
        y (np.ndarray): An array of integers [0, num_classes), of shape (n,)
        num_classes (int): Number of classes in y.

    Returns:
        np.ndarray: Array of shape (n, num_classes).
        One-hot representation of y (see below for example).

    Example:
        ```python
        > one_hot([2, 3, 1, 0], 4)
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        ```
    """
    one_hot = np.eye(len(set(y)))[y]

    return one_hot

def get_error(predictions, actual):
    # Compare the prediction with the acutal label.
    error = 0
    for i in range(len(predictions)):
        if predictions[i] != actual[i]:
            error = error + 1
    error = error / len(predictions)
    return error







def main():

    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    # Convert to one-hot
    y_train_one_hot = one_hot(y_train.reshape(-1), 10)
    _lambda = 1e-4

    w_hat = train(x_train, y_train_one_hot, _lambda)

    y_train_pred = predict(x_train, w_hat)
    y_test_pred = predict(x_test, w_hat)

    print("Ridge Regression Problem")
    print(
        f"\tTrain Error: {np.average(1 - np.equal(y_train_pred, y_train)) * 100:.5g}%"
    )
    print(f"\tTest Error:  {np.average(1 - np.equal(y_test_pred, y_test)) * 100:.5g}%")

    # p = 6000
    #
    # variance = 0.1
    #
    # shuffled_indices = np.arange(x_train.shape[0])
    # np.random.shuffle(shuffled_indices)
    # i = shuffled_indices[0:int(0.8 * x_train.shape[0])] # switch 0.8 to 1 to find the whole dataset
    #
    # Yp_train = y_train_one_hot[i, :]
    #
    #
    # G = np.random.normal(0, np.sqrt(variance), size=(p, x_train.shape[1]))
    # b = np.random.uniform(low=0, high=2 * np.pi, size=(p, 1))
    # X_T = np.cos(np.dot(x_train, G.T) + b.T)
    # Xp_train = X_T[i, :]
    # w_hat = train(Xp_train, Yp_train, _lambda)
    # X_test_transformed = np.cos(np.dot(x_test, G.T) + b.T)
    # Xtest = predict(X_test_transformed, w_hat)
    # test_error = get_error(Xtest, y_test)
    # square_root = np.sqrt(np.log(40) / (2 * (x_test.shape[0])))
    # print(test_error)
    # print(test_error - square_root)
    # print(test_error + square_root)


if __name__ == "__main__":
    main()
