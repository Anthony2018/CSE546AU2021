import typing_extensions

if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    # df_train has 96 features, make observations and targets
    x_train = df_train.drop(['ViolentCrimesPerPop'], axis = 1).values
    y_train = df_train['ViolentCrimesPerPop'].values
    x_test = df_test.drop(['ViolentCrimesPerPop'], axis = 1).values
    y_test = df_test['ViolentCrimesPerPop'].values

    # set the parameter
    ratio = 2
    end_point = 0.01
    feature = 95

    # inital the lambda and bias list
    lambda_list = []
    Bias = []

    # initial fist lambda, w
    lambda_i = lambda_max(x_train, y_train)
    weight = np.zeros((feature, ))
    W = np.zeros((feature, 1))
    nonzero = [ ]
    while lambda_i >= end_point:
        lambda_list.append(lambda_i)
        weight, bias = train(x_train, y_train, lambda_i, start_weight = weight)
        nonzero.append(np.count_nonzero(weight, axis = 0))
        W = np.concatenate((W, np.expand_dims(weight, axis = 1)), axis = 1)
        Bias.append(bias)
        weight = np.copy(weight)
        lambda_i = lambda_i / ratio
    # a3c
    plt.plot(lambda_list,nonzero,'o-')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Nonzeros')
    # plt.show()

    x_length = len(lambda_list)
    agePct12t29 = np.where(df_train.columns == 'agePct12t29')[0]
    pctWSocSec =  np.where(df_train.columns == 'pctWSocSec')[0]
    pctUrban = np.where(df_train.columns == 'pctUrban' )[0]
    agePct65up = np.where(df_train.columns == 'agePct65up')[0]
    householdsize = np.where(df_train.columns == 'householdsize')[0]

    #a3d
    plt.plot(lambda_list, np.reshape(W[agePct12t29-1,1:], (x_length,)), 'k^:' )
    plt.plot(lambda_list, np.reshape(W[pctWSocSec-1, 1:], (x_length,)), 'gv:')
    plt.plot(lambda_list, np.reshape(W[pctUrban-1, 1:], (x_length,)), 'b*-')
    plt.plot(lambda_list, np.reshape(W[agePct65up-1, 1:], (x_length,)), 'yx-')
    plt.plot(lambda_list, np.reshape(W[householdsize-1, 1:], (x_length,)), 'ro-')

    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Regularization paths for the coefficients')
    plt.legend(['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize'])
    # plt.show()

    #a3e
    y_train_pred = np.dot(W[:,1:].T, x_train.T) + np.expand_dims(Bias , axis = 1)
    y_test_pred = np.dot(W[:, 1:].T, x_test.T) + np.expand_dims(Bias, axis=1)

    train_error = square_loss(x_train.shape[0], y_train, y_train_pred)
    test_error = square_loss(x_test.shape[0], y_test, y_test_pred)



    plt.plot(lambda_list,train_error, 'ro-')
    plt.plot(lambda_list, test_error, 'b*-')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('square_loss')
    plt.legend(['train_error', 'test_error'])
    # plt.show()


    #a3f
    #set lamda = 30
    x_train = df_train.drop(['ViolentCrimesPerPop'], axis=1)
    lambda_i = 30
    weight = np.zeros((feature,))
    weight_30, bias_30 = train(x_train.values, y_train, lambda_i, start_weight=weight)

    print('Largest positive weight:', x_train.columns[np.argmax(weight_30)])
    print('Largest negative weight:', x_train.columns[np.argmin(weight_30)])






def lambda_max(X,y):
    # def the labmda with function
    lambda_max = 2 * np.max(np.abs(np.dot(y.T - np.mean(y), X)))
    return lambda_max


def square_loss(X,y, y_pre):
    return 1/X * np.sum(np.square(y_pre - y), axis = 1)







if __name__ == "__main__":
    main()
