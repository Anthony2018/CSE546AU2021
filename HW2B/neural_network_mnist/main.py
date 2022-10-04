# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot
import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.w0 = Parameter(torch.randn((h, d), dtype=torch.float32))
        self.w1 = Parameter(torch.randn((k, h), dtype=torch.float32))
        self.b0 = Parameter(torch.randn((1, h), dtype=torch.float32))
        self.b1 = Parameter(torch.randn((1, k), dtype=torch.float32))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        x = torch.mm(x, self.w0.T)+self.b0
        x = relu(x)
        x = torch.mm(x, self.w1.T)+self.b1
        return x


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.w0 = Parameter(torch.randn((h0, d), dtype=torch.float32))
        self.w1 = Parameter(torch.randn((h1, h0), dtype=torch.float32))
        self.w2 = Parameter(torch.randn((k, h1), dtype=torch.float32))

        self.b0 = Parameter(torch.randn((1, h0), dtype=torch.float32))
        self.b1 = Parameter(torch.randn((1, h1), dtype=torch.float32))
        self.b2 = Parameter(torch.randn((1, k), dtype=torch.float32))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        x = torch.mm(x, self.w0.T) + self.b0
        x = relu(x)
        x = torch.mm(x, self.w1.T) + self.b1
        x = relu(x)
        x = torch.mm(x, self.w2.T) + self.b2
        return x


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """

    model.train()
    train_loss = []
    test_loss = []
    train_err = []
    test_err = []

    train_err_epoch = 1
    device = 'cuda'
    model = model.to(device)
    loss_fun = cross_entropy
    count = 0
    while train_err_epoch > 0.01:
        train_err_epoch = 0
        loss_ls_epoch = 0
        test_err_epoch = 0
        test_loss_epoch = 0
        total_train = 0
        total_test = 0
        print('Epoch:', count)
        count += 1
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fun(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_hat = y_pred.max(dim=1)[1]
            err_train_batch = torch.count_nonzero(y_hat - y)
            train_err_epoch += err_train_batch
            loss_ls_epoch += loss.data.cpu()
            total_train +=len(y)
        train_err_epoch = train_err_epoch.cpu()/total_train
        train_loss.append(loss_ls_epoch.cpu()/total_train)
        train_err.append(train_err_epoch)
        print('train error', train_err_epoch)
        print('train loss', loss_ls_epoch.cpu()/total_train)

        # model.eval()
        #
        # for (x_test, y_test) in test_loader:
        #     x_test, y_test = x_test.to(device), y_test.to(device)
        #     y_test_pred = model(x_test)
        #     loss = cross_entropy(y_test_pred, y_test)
        #     y_hat_test = y_test_pred.max(dim=1)[1]
        #     err_train_batch = torch.count_nonzero(y_hat_test - y_test)
        #     test_err_epoch += err_train_batch
        #     test_loss_epoch += loss.data
        #     total_test += len(y_test)
        # test_loss.append(test_loss_epoch.cpu() / total_test)
        # test_err.append(test_err_epoch.cpu() / total_test)
        # print('test error', test_err_epoch.cpu() / total_test)
        # print('test loss', test_loss_epoch.cpu() / total_test)

    return train_loss


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    train_data = TensorDataset(x, y)
    test_data = TensorDataset(x_test, y_test)
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32, shuffle=False)
    h = 64
    d = 784
    k = 10
    h0 = 32
    h1 = 32
    model = F1(h, d, k)
    #model = F2(h0, h1, d, k)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    lr = 0.005
    device = 'cuda'
    optimizer = Adam(model.parameters(), lr=lr)
    #train_loss, train_err, test_loss, test_err = train(model, optimizer, trainloader, testloader)

    train_loss = train(model, optimizer, trainloader)

    plt.plot(train_loss, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch v.s. Loss')
    plt.legend(labels=['train_loss'])
    plt.show()


    # print the loss
    # plt.plot(train_loss, color='red')
    # plt.plot(test_loss, color='blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Epoch v.s. Loss')
    # plt.legend(labels=['train_loss', 'test_loss'])
    # plt.show()
    #
    # plt.plot(train_err, color='red')
    # plt.plot(test_err, color='blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('error')
    # plt.title('Epoch v.s. error')
    # plt.legend(labels=['train_error', 'test_error'])
    # plt.show()


if __name__ == "__main__":
    main()
