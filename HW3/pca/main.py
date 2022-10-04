from typing import Tuple

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a recontruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """
    projection = demean_data.dot(uk).dot(uk.T)
    return projection


@problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """
    projection = reconstruct_demean(uk, demean_data)
    error = np.mean(np.square(np.linalg.norm(demean_data - projection, axis=1)))
    return error


@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of it.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,)
            2. Matrix with eigenvectors as columns with shape (d, d)
    """

    demean_data_new = (demean_data.T).dot(demean_data)/len(demean_data)
    eig_list, eig_vector = np.linalg.eig(demean_data_new)
    return (np.real(eig_list), np.real(eig_vector))


@problem.tag("hw4-A", start_line=2)
def main():
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues

    Part C:
        - Plot reconstruction error as a function of k (# of eigenvectors used)
            Use k from 1 to 101.
            Plot should have two lines, one for train, one for test.
        - Plot ratio of sum of eigenvalues remaining after k^th eigenvalue with respect to whole sum of eigenvalues.
            Use k from 1 to 101.

    Part D:
        - Visualize 10 first eigenvectors as 28x28 grayscale images.

    Part E:
        - For each of digits 2, 6, 7 plot original image, and images reconstruced from PCA with
            k values of 5, 15, 40, 100.
    """

    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")
    # part a:
    x_tr_mu = np.mean(x_tr, axis=0)
    demean_x_tr = x_tr - x_tr_mu

    x_test_mu = np.mean(x_test, axis=0)
    demean_x_test = x_test - x_test_mu


    eig_values,_ = calculate_eigen(demean_x_tr)
    lam = np.sort(eig_values)[::-1]
    lam_list = [1, 2, 10, 30, 50]
    for i in lam_list:
        print("lambda",str(i),' is', lam[i-1])
    print('lambda sum is', np.sum(lam))
    #
    #part c:
    k_list = np.arange(1, 101)
    eig_tr, eigv_tr = calculate_eigen(demean_x_tr)
    eig_test, eigv_test =calculate_eigen(demean_x_test)
    train_error_ls = []
    test_error_ls = []
    ref_ls = []
    for k in k_list:
        train_error = reconstruction_error(eigv_tr[:, 0:k], demean_x_tr)
        train_error_ls.append(train_error)
        test_error = reconstruction_error(eigv_tr[:, 0:k], demean_x_test)
        test_error_ls.append(test_error)
        ref = 1 - np.sum(eig_tr[0:k] / np.sum(eig_tr))
        ref_ls.append(ref)
    plt.plot(train_error_ls, 'r')
    plt.plot(test_error_ls, 'b')
    plt.legend(['reconstruction train error','reconstruction test error'])
    plt.title('reconstruction error on train and test sets')
    plt.xlabel('k')
    plt.ylabel('mean-squared loss')
    plt.show()
    plt.plot(ref_ls)
    plt.xlabel('k')
    plt.ylabel('1- lam(1:k)/sum(lam)')
    plt.title('k v.s. 1- lam(1:k)/sum(lam)')

    plt.show()

    # d
    eig_values, eigv_values = calculate_eigen(demean_x_tr)
    top = eig_values.argsort()[-10:][::-1]

    plt.figure(12)
    for i in range(10):
        print(i)
        plt.subplot(2, 5, int(i+1))
        plt.imshow(eigv_values[:, top[i]].reshape(28,28))
        plt.title('top '+str(i+1)+' eig_v')
    plt.show()


    # e
    k_list = [5, 15, 40, 100]
    eig_values, eigv_values = calculate_eigen(demean_x_tr)
    i =1
    for k in k_list:
        plt.subplot(2,3,i)
        plt.imshow(reconstruct_demean( eigv_values[:, 0:k], demean_x_tr[5].reshape(1,-1),).reshape(28, 28))
        plt.title('reconstruct 2, k = '+str(k))
        i +=1

    plt.subplot(2, 3, 5)
    plt.imshow(demean_x_tr[5].reshape(28, 28))
    plt.title('plot 2')
    matplotlib.pyplot.show()


    i = 1
    for k in k_list:
        plt.subplot(2, 3, i)
        plt.imshow(reconstruct_demean(eigv_values[:, 0:k], demean_x_tr[13].reshape(1, -1), ).reshape(28, 28))
        plt.title('reconstruct 6, k = ' + str(k))
        i += 1

    plt.subplot(2, 3, 5)
    plt.imshow(demean_x_tr[13].reshape(28, 28))
    plt.title('plot 6')
    matplotlib.pyplot.show()

    i = 1
    for k in k_list:
        plt.subplot(2, 3, i)
        plt.imshow(reconstruct_demean(eigv_values[:, 0:k], demean_x_tr[15].reshape(1, -1), ).reshape(28, 28))
        plt.title('reconstruct 7, k = ' + str(k))
        i += 1

    plt.subplot(2, 3, 5)
    plt.imshow(demean_x_tr[15].reshape(28, 28))
    plt.title('plot 7')
    matplotlib.pyplot.show()


    k_list = [32, 64, 128]
    eig_tr, eigv_tr = calculate_eigen(demean_x_tr)
    eig_test, eigv_test = calculate_eigen(demean_x_test)
    train_error_ls = []
    test_error_ls = []
    for k in k_list:
        train_error = reconstruction_error(eigv_tr[:, 0:k], demean_x_tr)
        train_error_ls.append(train_error)
        test_error = reconstruction_error(eigv_tr[:, 0:k], demean_x_test)
        test_error_ls.append(test_error)
        print(train_error, test_error)







if __name__ == "__main__":
    main()
