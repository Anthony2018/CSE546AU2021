import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


@problem.tag("hw4-A")
def F1(h: int) -> nn.Module:
    """Model F1, it should performs an operation W_d * W_e * x as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    model = nn.Sequential(
        torch.nn.Linear(784,h),
        torch.nn.Linear(h,784)
    )
    return model.cuda()




@problem.tag("hw4-A")
def F2(h: int) -> nn.Module:
    """Model F1, it should performs an operation ReLU(W_d * ReLU(W_e * x)) as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    model = nn.Sequential(
        torch.nn.Linear(784, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 784),
        torch.nn.ReLU(),
    )
    return model.cuda()


@problem.tag("hw4-A")
def train(
    model: nn.Module, optimizer: Adam, train_loader: DataLoader, epochs: int = 40
) -> float:
    """
    Train a model until convergence on train set, and return a mean squared error loss on the last epoch.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
            Hint: You can try using learning rate of 5e-5.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Note:
        - Unfortunately due to how DataLoader class is implemented in PyTorch
            "for x_batch in train_loader:" will not work. Use:
            "for (x_batch,) in train_loader:" instead.

    Returns:
        float: Final training error/loss
    """
    model.train()
    train_loss = []
    device = 'cuda'
    loss_fun = nn.MSELoss()

    for i in range(epochs):
        loss_ls_epoch = 0
        total_train = 0
        for (x,) in train_loader:
            x = x.to(device)
            y_pred = model(x)
            loss = loss_fun(y_pred, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ls_epoch += loss.data.cpu()
            total_train += len(x)
        train_loss = (loss_ls_epoch / total_train)
        print('the epoch = '+str(i)+' ,train loss is ', str(train_loss))
    return train_loss


@problem.tag("hw4-A")
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluates a model on a provided dataset.
    It should return an average loss of that dataset.

    Args:
        model (Module): TRAINED Model to evaluate. Either F1, or F2 in this problem.
        loader (DataLoader): DataLoader with some data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Returns:
        float: Mean Squared Error on the provided dataset.
    """
    model.eval()
    device = 'cuda'
    loss_fun = nn.MSELoss()
    total_test = 0
    test_loss_epoch = 0

    for (x_test,) in loader:

        x_test = x_test.to(device)
        y_test_pred = model(x_test)
        loss = loss_fun(y_test_pred, x_test)

        test_loss_epoch += loss.data
        total_test += len(x_test)

    test_loss = test_loss_epoch.cpu() / total_test
    print('test loss', test_loss)

    return test_loss


@problem.tag("hw4-A", start_line=9)
def main():
    """
    Main function of autoencoders problem.

    It should:
        A. Train an F1 model with hs 32, 64, 128, report loss of the last epoch
            and visualize reconstructions of 10 images side-by-side with original images.
        B. Same as A, but with F2 model
        C. Use models from parts A and B with h=128, and report reconstruction error (MSE) on test set.

    Note:
        - For visualizing images feel free to use images_to_visualize variable.
            It is a FloatTensor of shape (10, 784).
        - For having multiple axes on a single plot you can use plt.subplots function
        - For visualizing an image you can use plt.imshow (or ax.imshow if ax is an axis)
    """
    (x_train, y_train), (x_test, _) = load_dataset("mnist")
    x = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    # Neat little line that gives you one image per digit for visualization in parts a and b
    images_to_visualize = x[[np.argwhere(y_train == i)[0][0] for i in range(10)]]
    for i in range(len(images_to_visualize)):
        plt.subplot(2, 5, i+1)
        plt.imshow(images_to_visualize[i].reshape(28, 28))
    plt.show()

    train_loader = DataLoader(TensorDataset(x), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=32, shuffle=True)

    h_list = [32, 64, 128]
    for h in h_list:
        model = F2(h)
        optimizer = Adam(model.parameters(), lr=0.0005)
        train_loss = train(model, optimizer, train_loader)
        test_loss = evaluate(model, test_loader)
        print('h = '+str(h)+' the train loss is '+str(train_loss))
        print('h = ' + str(h) + ' the test loss is ' + str(test_loss))
        reconstruction = model(images_to_visualize.cuda()).cpu().detach().numpy()
        counter = 1
        plt.figure(figsize=(12, 12), dpi=80)

        for i in range(len(images_to_visualize)):
            plt.subplot(4, 5, counter)
            plt.imshow(images_to_visualize[i].reshape(28, 28))
            plt.title('digit'+str(i))
            counter += 1
            plt.subplot(4, 5, counter)
            plt.imshow(reconstruction[i].reshape(28, 28))
            plt.title('reconstruction digit ' + str(i))
            counter += 1
        plt.show()



if __name__ == "__main__":
    main()
