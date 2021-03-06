import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def partial_derivatives(X, beta, alpha):
    X = np.array(X)
    n = X.shape[0]
    # derivative of Likely-Hood function with respect to alpha
    m_gradient = (n / beta) - (sum(np.exp(-(X - alpha / beta))) / beta)
    # derivative of Likely-Hood function with respect to beta
    b_gradient = - (n / beta) + sum((X - alpha) / beta ** 2) - np.dot(((X - alpha) / beta ** 2),
                                                                      np.exp(-(X - alpha / beta)))
    return b_gradient, m_gradient


def estimating_parameters(X, epochs, gema):
    precision = 0.001  # Desired precision of result
    # initializing alpha and beta with method of moment
    beta = (np.std(X) * np.sqrt(6)) / np.pi
    alpha = np.mean(X) - (0.57721 * beta)

    # applying gradient Accent
    for i in range(epochs):

        current_beta = beta
        current_mu = alpha
        # getting partial derivative
        b_gradient, m_gradient = partial_derivatives(X, beta, alpha)
        # maximizing our likely_hood function
        beta = beta + (gema * b_gradient)
        alpha = alpha + (gema * m_gradient)
        # early stopping
        beta_diff = beta - current_beta
        mu_diff = alpha - current_mu
        if abs(mu_diff) <= precision and abs(beta_diff) <= precision:
            break

    return beta, alpha


def visualization(data, beta , alpha, dis):
    # getting parameters with the help of moment generating function
    b = (np.std(data) * np.sqrt(6)) / np.pi
    a = np.mean(data) - (0.57721 * beta)

    # getting parameters with help of parameters estimations
    b_prediction, a_prediction = estimating_parameters(data, max_iters, gema) #

    # visulization
    count, bins, ignored = plt.hist(data, 30, density=True)
    # plt original KDE
    plt.plot(bins, (1 / beta) * np.exp(-(bins - alpha) / beta)
             * np.exp(-np.exp(-(bins - alpha) / beta)),
             linewidth=2, label='Original KDE', color='g')
    # plt method of moment/ predicted KDE
    plt.plot(bins, (1 / b) * np.exp(-(bins - m) / b)
             * np.exp(-np.exp(-(bins - a) / b)),
             linewidth=2, label='Moment KDE', color='r')
    plt.plot(bins, (1 / b_prediction) * np.exp(-(bins - a_prediction) / b_prediction)
             * np.exp(-np.exp(-(bins - a_prediction) / b_prediction)),
             linewidth=2, label='Predicted KDE', color='y')
    plt.title('Histogram of the Samples({})'.format(dis))
    plt.xlabel('Bins')
    plt.ylabel('Count')
    plt.legend(loc="upper right")
    # plt.savefig('image_name.png')
    plt.show()


if __name__ == "__main__":
    alpha, beta = 1, 2  # location and scale
    n = [100, 1000, 10000]  # add 10000
    max_iters = 1000  # Maximum number of iterations
    gema = 0.001

    for value in n:
        alpha_list = list()
        beta_list = list()
        mean_list = list()
        std_list = list()
        # create 10 set for every n and randomly create gumbel distribution
        dataset = [np.random.gumbel(alpha, beta, value) for j in range(10)]
        # for loop for every data in our dataset
        for data in dataset:
            # estimating parameters
            b, m = estimating_parameters(data, max_iters, gema)
            mean = m + (0.57721 * beta)
            std = (beta * np.pi) / (np.sqrt(6))
            alpha_list.append(m)
            beta_list.append(b)
            mean_list.append(mean)
            std_list.append(std)
        df = pd.DataFrame({"Alpha": alpha_list, "Beta": beta_list, "Mean_signle_Data":mean_list, "std_signle_Data":std_list})
        df = df.T
        print("Table for {}".format(value))
        print(df)
        visualization(dataset[1], beta, alpha, value)