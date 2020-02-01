import numpy as np


def partial_derivatives(X,beta,alpha):
    X = np.array(X)
    n = X.shape[0]
    m_gradient = (n/beta) - (sum(np.exp(-(X - alpha / beta))) / beta)
    b_gradient = - (n/beta) + sum((X - alpha) / beta ** 2) - np.dot(((X - alpha) / beta ** 2), np.exp(-(X - alpha / beta)))
    return b_gradient,m_gradient


def estimating_parameters(X, epochs, alpha):
    precision = 0.001  # Desired precision of result
    beta = (np.std(X) * np.sqrt(6)) / np.pi
    mu = np.mean(X) - (0.57721 * beta)

    for i in range(epochs):
        current_beta = beta
        current_mu = mu
        b_gradient, m_gradient = partial_derivatives(X, beta, mu)
        beta = beta + (gema * b_gradient)
        mu = mu + (gema * m_gradient)
        beta_diff = beta - current_beta
        mu_diff = mu - current_mu
        if abs(mu_diff) <= precision and abs(beta_diff) <= precision:
            break
    return beta, mu


if __name__ == "__main__":
    mu, beta = 3, 2  # location and scale
    n = [100, 1000] # add 10000
    max_iters = 1000  # Maximum number of iterations
    gema = 0.001
    # create 10 set for every n and randomly create gumbel distribution
    data = [np.random.gumbel(mu, beta, j) for i in range(10) for j in n]
    for d in data:
        b, m = estimating_parameters(d, max_iters, gema)
        mean = m + (0.57721 * beta)
        std = (beta * np.pi) / (np.sqrt(6))
        print("Mean : {0}  Std : {1} α : {2}  β : {3}".format(mean, std, m, b))