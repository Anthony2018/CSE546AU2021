if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


RNG = torch.Generator()
RNG.manual_seed(446)

class Linear_Regression_Model(nn.Module):
    def __init__(self):
        super(Linear_Regression_Model, self).__init__()
        self.layer = LinearLayer(2, 2)

    def forward(self, x):
        x = self.layer(x)
        return x

class Linear_Model_sigmoid(nn.Module):
    def __init__(self):
        super(Linear_Model_sigmoid, self).__init__()
        self.layer = LinearLayer(2, 2)
        self.layer_h = LinearLayer(2, 2)
        self.sigmoid = SigmoidLayer()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x

class Linear_Model_relu(nn.Module):
    def __init__(self):
        super(Linear_Model_relu, self).__init__()
        self.layer = LinearLayer(2, 2)
        self.layer_h = LinearLayer(2, 2)
        self.relu = ReLULayer()

    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        return x

class Linear_Model_relu_sigmod(nn.Module):
    def __init__(self):
        super(Linear_Model_relu_sigmod, self).__init__()
        self.layer = LinearLayer(2, 2)
        self.layer_h1 = LinearLayer(2, 2)
        self.layer_h2 = LinearLayer(2, 2)
        self.relu = ReLULayer()
        self.sigmoid = SigmoidLayer()

    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        x = self.layer_h1(x)
        x = self.sigmoid(x)
        return x

class Linear_Model_sigmod_relu(nn.Module):
    def __init__(self):
        super(Linear_Model_sigmod_relu, self).__init__()
        self.layer = LinearLayer(2, 2)
        self.layer_h1 = LinearLayer(2, 2)
        self.layer_h2 = LinearLayer(2, 2)
        self.relu = ReLULayer()
        self.sigmoid = SigmoidLayer()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        x = self.layer_h1(x)
        x = self.relu(x)
        return x


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    model.eval()
    accuracy = 0
    y_val_len = 0
    for (x_val, y_val) in dataloader:
        pred_val = model(x_val)
        pred = pred_val.argmax(dim=1, keepdim=True).numpy()
        pred = np.squeeze(pred)
        one_hot = torch.from_numpy(to_one_hot(pred))
        accuracy += one_hot.eq(y_val.view_as(one_hot)).sum().item()
        y_val_len += len(y_val)

    return accuracy/y_val_len/2


@problem.tag("hw3-A")
def mse_parameter_search(
    dataloader_train: DataLoader, dataloader_val: DataLoader
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Args:
        dataloader_train (DataLoader): Dataloader for training dataset.
        dataloader_val (DataLoader): Dataloader for validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    name_model = ['Linear_Regression_Model', 'Linear_Model_sigmoid', 'Linear_Model_relu', 'Linear_Model_relu_sigmod',
                  'Linear_Model_sigmod_relu']
    models = [Linear_Regression_Model(), Linear_Model_sigmoid(), Linear_Model_relu(), Linear_Model_relu_sigmod(),
              Linear_Model_sigmod_relu()]
    dic_ls = {}
    count = 0
    for model in models:
        # model.to('cuda')
        optimizer = SGDOptimizer(model.parameters(), lr=0.03)
        dic = train(dataloader_train, model, MSELossLayer(), optimizer, dataloader_val)
        dic_data = {'train': dic['train'], 'val': dic['val'], 'model': model}
        dic_ls[name_model[count]] = dic_data
        count += 1
    return dic_ls



@problem.tag("hw3-A", start_line=21)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    mse_dataloader_train = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y))),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )
    mse_dataloader_val = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
        ),
        batch_size=32,
        shuffle=False,
    )
    mse_dataloader_test = DataLoader(
        TensorDataset(
            torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
        ),
        batch_size=32,
        shuffle=False,)
    color_set = ['ro-', 'bo-', 'rv:', 'bv:', 'r*-', 'b*-', 'rx-', 'bx-', 'r', 'b']
    count = 0
    labels = []
    ce_configs = mse_parameter_search(mse_dataloader_train, mse_dataloader_val)
    for key, value in ce_configs.items():
        plt.plot(value['train'], color_set[count])
        plt.plot(value['val'], color_set[count + 1])
        labels.append(key + '  train loss')
        labels.append(key + '  val_loss')
        count = count + 2

    plt.legend(labels)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('5 NNs train loss and val loss v.s. iterations (MSE)')
    plt.show()

    for key, value in ce_configs.items():
        model = value['model']
        print('val error for ', key, ' is', accuracy_score(model, mse_dataloader_val))
        print('train error for ', key, ' is', accuracy_score(model, mse_dataloader_train))
        print('test error for ', key, ' is', accuracy_score(model, mse_dataloader_test))
        plot_model_guesses(mse_dataloader_test, model)


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
