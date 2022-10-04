if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem
import pickle


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    You should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")
    # k = 10
    # error_list, classification, centers = lloyd_algorithm(x_train, k)
    # plt.plot(error_list)
    # plt.xlabel('iterations')
    # plt.ylabel('objective function loss')
    # plt.title('objective function')
    # plt.show()
    #
    # for i in range(k):
    #     plt.imshow(centers[i].reshape(28, 28))
    #     plt.show()
    k_list = [2, 4, 8, 16, 32, 64]
    # error_dic = dict()
    # class_dic = dict()
    # centers_dic = dict()
    # for k in k_list:
    #     print('calculate the k = ',str(k))
    #     error_list, classification, centers = lloyd_algorithm(x_train, k)
    #     error_dic[k] = error_list
    #     class_dic[k] = classification
    #     centers_dic[k] = centers
    # with open('class_dic.pickle', 'wb') as p:
    #     pickle.dump(class_dic, p)
    # with open('center_dic.pickle', 'wb') as p:
    #     pickle.dump(centers_dic, p)
    # with open('error.pickle', 'wb') as p:
    #     pickle.dump(error_dic, p)


    # with open('class_dic.pickle', 'rb') as p:
    #     class_dic =  pickle.load(p)
    with open('center_dic.pickle', 'rb') as p:
        centers_dic = pickle.load(p)
    with open('error.pickle', 'rb') as p:
        error_dic = pickle.load(p)

    train_error = []
    test_error = []
    for k in k_list:
        print('calculate the test error, k = ', str(k))
        test_error_k = calculate_error(x_test, centers_dic[k])
        test_error.append(test_error_k)
        train_error_k = error_dic[k][-1]
        train_error.append(train_error_k)
        print('test error, k = ', str(k), test_error_k)
        print('train error, k = ', str(k), train_error_k)

    plt.plot(k_list, train_error,  color='red', label ='train error')
    plt.plot(k_list, test_error, color='blue', label='test error')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('training and test error as function of k')
    plt.show()


if __name__ == "__main__":
    main()
