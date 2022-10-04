import matplotlib.pyplot as plt
import numpy as np

from utils import problem


def main():
    required_std = 0.0025
    n = int(np.ceil(1.0 / (required_std * 2))) ** 2
    ks = [1, 8, 64, 512]
    for k in ks:
        Y_k = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1.0 / k), axis=1)
        plt.step(sorted(Y_k), np.arange(1, n + 1) / float(n), label=str(k))

    # Plot gaussian
    Z = np.random.randn(n)
    plt.step(sorted(Z), np.arange(1, n + 1) / float(n), label="Gaussian")
    plot_settings()


@problem.tag("hw0-A", start_line=8)
def plot_settings():
    # Plotting settings
    plt.grid(which="both", linestyle="dotted")
    plt.legend(
        loc="lower right", bbox_to_anchor=(1.1, 0.2), fancybox=True, shadow=True, ncol=1
    )
    # TODO: Look through matplotlib documentation and:
    #   - limit x axis to be between -3 and 3
    #   - Add label "Observations" on x axis
    #   - Add label "Probability" on y axis

    ax = plt.gca()
    plt.title("Empirical CDF of mean of N(0,1) variables (n=40000)")
    ax.set_xlim((-3, 3))
    plt.xlabel('Observations')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
