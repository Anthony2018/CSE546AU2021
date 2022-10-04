"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # # You can add additional fields
        # raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        A = X
        for d in range(1, degree):
            d += 1
            A = np.concatenate((A, X**d), axis = 1)
        return A

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        # X (n,1) to A (n,d)
        A = self.polyfeatures(X , self.degree)

        # norm

        if X.shape[0] == 1:
            self.mean = 0
            self.std = 1
        else:
            self.mean = A.mean(axis = 0)
            self.std = A.std(axis = 0)

        A = (A - self.mean) / self.std

        # add ones
        A = np.concatenate((np.ones((X.shape[0],1)),A), axis = 1)

        # reg matrix
        reg_m = self.reg_lambda * np.eye(A.shape[1])
        reg_m[0,0] = 0

        self.weight = np.linalg.pinv(A.T.dot(A) + reg_m).dot(A.T).dot(y)

        return self.weight



    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        pred_expand = self.polyfeatures(X, self.degree)
        pred_std = (pred_expand - self.mean) / self. std
        # add ones
        pred_ones = np.concatenate((np.ones((X.shape[0],1)),pred_std), axis = 1)
        pred = pred_ones.dot(self.weight)
        return  pred


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    return np.mean(np.square(a - b))


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    test_x = Xtest
    test_y = Ytest
    for i in range(1,n):
        train_x = Xtrain[0:i+1]
        train_y = Ytrain[0:i+1]

        pred_model = PolynomialRegression(degree, reg_lambda)
        pred_model.fit(train_x, train_y)

        pred_train = pred_model.predict(train_x)
        pred_test  = pred_model.predict(test_x)

        train_err = mean_squared_error(pred_train,train_y)
        test_err = mean_squared_error(pred_test,test_y)

        errorTrain[i] = train_err
        errorTest[i] = test_err

    return errorTrain, errorTest
