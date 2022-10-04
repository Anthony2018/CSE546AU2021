if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
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
        self.softmax = SoftmaxLayer()

    def forward(self, x):
        x = self.layer(x)
        x = self.softmax(x)
        return x

class Linear_Model_sigmoid(nn.Module):
    def __init__(self):
        super(Linear_Model_sigmoid, self).__init__()
        self.layer = LinearLayer(2, 2)
        self.sigmoid = SigmoidLayer()
        self.softmax = SoftmaxLayer()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        return x

class Linear_Model_relu(nn.Module):
    def __init__(self):
        super(Linear_Model_relu, self).__init__()
        self.layer = LinearLayer(2, 2)
        self.softmax = SoftmaxLayer()
        self.relu = ReLULayer()

    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x

class Linear_Model_relu_sigmod(nn.Module):
    def __init__(self):
        super(Linear_Model_relu_sigmod, self).__init__()
        self.layer = LinearLayer(2, 2)
        self.layer_h1 = LinearLayer(2, 2)
        self.softmax = SoftmaxLayer()
        self.relu = ReLULayer()
        self.sigmoid = SigmoidLayer()

    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        x = self.layer_h1(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        return x

class Linear_Model_sigmod_relu(nn.Module):
    def __init__(self):
        super(Linear_Model_sigmod_relu, self).__init__()
        self.layer = LinearLayer(2, 2)
        self.layer_h1 = LinearLayer(2, 2)
        self.softmax = SoftmaxLayer()
        self.relu = ReLULayer()
        self.sigmoid = SigmoidLayer()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        x = self.layer_h1(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x

@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataloader_train: DataLoader, dataloader_val: DataLoader
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

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
    name_model = ['Linear_Regression_Model', 'Linear_Model_sigmoid', 'Linear_Model_relu', 'Linear_Model_relu_sigmod', 'Linear_Model_sigmod_relu']
    models = [Linear_Regression_Model(), Linear_Model_sigmoid(), Linear_Model_relu(), Linear_Model_relu_sigmod(), Linear_Model_sigmod_relu()]

    dic_ls = {}
    count = 0
    for model in models:
        # model.to('cuda')
        optimizer = SGDOptimizer(model.parameters(), lr=0.03)
        dic = train(dataloader_train, model, CrossEntropyLossLayer(), optimizer, dataloader_val)
        dic_data = {'train': dic['train'], 'val': dic['val'], 'model': model}
        dic_ls[name_model[count]] = dic_data
        count += 1
    return dic_ls


@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    model.eval()
    accuracy = 0
    y_val_len = 0
    for (x_val, y_val) in dataloader:

        pred_val = model(x_val)
        pred = pred_val.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(y_val.view_as(pred)).sum().item()
        y_val_len += len(y_val)

    return accuracy/y_val_len


@problem.tag("hw3-A", start_line=21)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    ce_dataloader_train = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y)),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )
    ce_dataloader_val = DataLoader(
        TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val)),
        batch_size=32,
        shuffle=False,
    )
    ce_dataloader_test = DataLoader(
        TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test)),
        batch_size=32,
        shuffle=False,
    )

    ce_configs = crossentropy_parameter_search(ce_dataloader_train, ce_dataloader_val)

    color_set=['ro-', 'bo-', 'rv:', 'bv:', 'r*-', 'b*-', 'rx-', 'bx-', 'r', 'b']
    count = 0
    labels = []
    for key, value in ce_configs.items():
        plt.plot(value['train'], color_set[count])
        plt.plot(value['val'], color_set[count+1])
        labels.append(key+'  train loss')
        labels.append(key+'  val_loss')
        count = count + 2

    plt.legend(labels)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('5 NNs train loss and val loss v.s. iterations (crossentropy search)')
    plt.show()

    for key, value in ce_configs.items():
        model = value['model']
        print('val error for ', key, ' is', accuracy_score(model, ce_dataloader_val))
        print('train error for ', key, ' is', accuracy_score(model, ce_dataloader_train))
        print('test error for ', key, ' is', accuracy_score(model, ce_dataloader_test))
        plot_model_guesses(ce_dataloader_test, model)




if __name__ == "__main__":
    main()
