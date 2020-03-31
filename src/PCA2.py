"""
Principal Component Analysis (PCA)

Author: Zohreh Raziei - raziei.z@husky.neu.edu
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp



class logistic_regression:
    """
    Class implements logistic_regression algorithm.
    """

    def __init__(self, solver='gradient_descent', variant='batch', alpha=0.01, tol=0.0001, max_iter=1000):

        """

        Constructor of logistic_regression.

        Arguments:

        1. solver : This will takes two values  "gradient_descent" and "Newton". We will select in which way we
        we want to solve our problem.
        2. variant: If we select the gradient_descent then we want to apply gradient_descent. We have two choices "batch" and
        "stochastic". For stochastic gradient descent in which we select one example at a time and
         perform the optimization step in logistic regression and in batch we will perform optimization on whole dataset
        3. Alpha: alpha is our learning rate which we use for gradinet descent.
        4. Tolerance: This is the stoping condition when our change_cost is less than tolerance then algorithm stops processing.
        5. Max iteration: How many iteration we want to perform.

        """

        self.__solver = solver
        self.__variant = variant
        self.__tol = tol
        self.__max_iter = max_iter
        self.__alpha = alpha
        self.__X_training = None
        self.__Y_training = None
        self.__theta = None  # this will be our w1,w2 etc according to our features

    def __sigmoid_func(self, X):

        """!
        This function is used to predict the probabilites of classes.

        Argument:
        1. X: X is numpy array with all examples and features.

        Return:
              result: sigmoid function output
        """
        # find dot product with training data and theta (w1,w2 etc)
        dot_product = X.dot(self.__theta)
        # create empty result list
        result = np.zeros(dot_product.shape)
        # split into positive and negative to improve stability
        result[dot_product >= 0.0] = 1.0 / (1.0 + np.exp(-dot_product[dot_product >= 0.0]))
        result[dot_product < 0.0] = np.exp(dot_product[dot_product < 0.0]) / (
                    np.exp(dot_product[dot_product < 0.0]) + 1.0)

        return result

    def __cost_func(self):
        """
        This function is used to compute negative_log-likelihood or simply cost function

        Return:
        This will return cost value how much we predicted wrong.
        """
        # calling sigmoid_func with training data to calculate the predictions
        result = self.__sigmoid_func(self.__X_training)
        # apply negative log-likelihood function
        positive_examples = sum(np.log(result[self.__Y_training > 0.5]++0.00001)) # add some small number to avoid 0 values in log
        negative_examples = sum(np.log(1 - result[self.__Y_training < 0.5]+0.00001))
        return -positive_examples - negative_examples

    def __neg_log_grad(self, X, Y):
        """
        This function is used to compute the derivative value of the negative log-likelihood.

        Return:
             derivative value of negative log-likelihood
        """
        result = self.__sigmoid_func(X)
        return -X.T.dot(Y - result)

    def __grad_desc(self):
        """
        This function is used for the gradient descent.

        Return:
        How many iteration is performed for converges.
        """
        # calculate the cost and append in the list
        cost = [self.__cost_func()]
        change_cost = 2.0 * self.__tol
        iterations = 0
        # stoping condtions
        while (change_cost > self.__tol) and (iterations < self.__max_iter):
            # apply gradinet
            self.__theta = self.__theta - (self.__alpha * self.__neg_log_grad(self.__X_training, self.__Y_training))
            # calculate the cost again
            cost.append(self.__cost_func())
            # check the cost b/w last cost and second last cost.
            change_cost = cost[-2] - cost[-1]
            iterations += 1
        return iterations

    def __stochastic_grad_desc(self):

        """
        This function is used for the stochastic gradient descent.

        Return:
        How many iteration is performed for converges.
        """
        # calculate the cost and append in the list
        cost = [self.__cost_func()]
        change_cost = 2.0 * self.__tol
        iterations = 0
        # stoping condtions
        while (change_cost > self.__tol) and (iterations < self.__max_iter):
            # perform gradient descent for single recode.
            for row in range(len(self.__X_training)):
                # apply gradinet
                self.__theta = self.__theta - (
                            self.__alpha * self.__neg_log_grad(self.__X_training[row], self.__Y_training[row]))
            cost.append(self.__cost_func())
            # check the cost b/w last cost and second last cost.
            change_cost = cost[-2] - cost[-1]
            iterations += 1

        return iterations

    def __log_hessian(self):
        """
        This function is used to compute the hessian of the negative log-likelihood

        Return:

        This will return hessian matrix (m*m)
        """
        x = self.__X_training[:, np.newaxis]
        result = self.__sigmoid_func(self.__X_training)
        result_3d = result[:, np.newaxis, np.newaxis]
        # apply hessian formula
        hessian = np.sum(x.transpose((0, 2, 1)) * x * (result_3d * (1 - result_3d)), axis=0)
        return hessian

    def __newton_method(self):
        """
        This function is used for the newton_method.

        Return:
        How many iteration is performed for converges.
        """
        cost = [self.__cost_func()]
        change_cost = 2.0 * self.__tol
        iterations = 0
        # stoping condtions
        while (change_cost > self.__tol) and (iterations < self.__max_iter):
            hessian = self.__log_hessian()
            # calculate inverse matrix of hessian
            hessian_inv = np.linalg.inv(hessian)
            # apply newton method formula
            self.__theta = self.__theta - (hessian_inv.dot(self.__neg_log_grad(self.__X_training, self.__Y_training)))
            cost.append(self.__cost_func())
            change_cost = cost[-2] - cost[-1]
            iterations += 1

        return iterations

    def fit(self, X, Y):
        """
        This function is used to train the model and solve the problem according to given conditions
        """
        self.__Y_training = Y
        shape = X.shape
        # We will add One column with values 1. because for linear regration w1*x1+b*X0. So we will set X0 is equal to one.
        # so we can use numpy for faster oprations
        self.__X_training = np.zeros((shape[0], shape[1] + 1))
        self.__X_training[:, 0] = np.ones(shape[0])
        self.__X_training[:, 1:] = X
        # Initialize theta to zeros
        self.__theta = np.zeros(shape[1] + 1)

        if self.__solver == "gradient_descent":
            if self.__variant == "batch":
                iterations = self.__grad_desc()
            else:
                iterations = self.__stochastic_grad_desc()
        else:
            iterations = self.__newton_method()

    # function to compute output of LR classifier
    def predict(self, X):

        """
        This function is used to predict the Values of new data.

        Return:
        This will return the predict class of new data.
        """
        # Same as above
        shape = X.shape
        self.__X_testing = np.zeros((shape[0], shape[1] + 1))
        self.__X_testing[:, 0] = np.ones(shape[0])
        self.__X_testing[:, 1:] = X
        # predict the probabilites of classes
        pred_prob = self.__sigmoid_func(self.__X_testing)
        # convert probabilies into 1 and 0
        pred_value = np.where(pred_prob >= .5, 1, 0)

        return np.squeeze(pred_value)

    def score(self, X, y):
        """
        This function will predict the Scores or accuray of model
        """
        return sum(self.predict(X) == y) / len(y)


class PCA:
    """
    Class implements PCA algorithm.
    """

    def __init__(self, n_components=2, retained_variance=True):
        """!
        Constructor of PCA.

        Arguments:
        n_components: how many features we want
        retained_variance: if retained_variance is true then we will have to find best k features which gets 99% variance.
        """
        self.__n_components = n_components
        self.__retained_variance = retained_variance
        self.__eigvalues = None
        self.__eigvectors = None

    def fit(self, X):
        """
        This function will find the best k features if retained_variance and calculte eigvalues and eigvectores.
        """
        covar_matrix = np.cov(X.T)
        self.__eigvalues, self.__eigvectors = np.linalg.eig(covar_matrix)
        if self.__retained_variance == True:
            for i in range(self.__eigvalues.shape[0]):
                variance = sum(self.__eigvalues[:i + 1]) / sum(self.__eigvalues)
                if variance >= 0.99:
                    self.__n_components = i
                    break

    def transform(self, X):
        """
        This function will gives us k features aftter transforming
        """
        top_k_eigvectors = self.__eigvectors[:, 0:self.__n_components]
        return X.dot(top_k_eigvectors)


class Normalization:
    """
        Class implements Normalization.
        """

    def __init__(self, normalization="Z_score"):
        """!
                Constructor of Normalization.

                Arguments:
                normalization: what type of normalization we want, "Z_score" or "Zero_mean".
                """
        self.__normalization = normalization
        self.__means = None
        self.__stds = None

    def __zero_mean(self, X):
        """
        This function subtract means from features
        """
        norm_X = X - self.__means
        return norm_X

    def __z_score(self, X):
        """
        This function implement Z_Score formula
        """
        norm_X = (X - self.__means) / self.__stds
        return norm_X

    def fit(self, X):
        """
        This function is used to calculate means and std of our every features and save it
        """
        self.__means = np.mean(X, axis=0)
        self.__stds = np.std(X, axis=0)

    def transform(self, X):
        """
        This function check the condition and call the normalization function
        Return:
            this will return normalize data.
        """
        if self.__normalization == "Z_score":
            return self.__z_score(X)
        else:
            return self.__zero_mean(X)


def k_fold_validation(kf,model, data,Y):
    cv_score = []
    i = 1
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1., 0.1))
    for train_index, test_index in kf.split(data, Y):
        print('{} of KFold {}'.format(i, kf.n_splits))
        x_tr, x_l = data.iloc[train_index], data.iloc[test_index]
        y_tr, y_vl = Y.iloc[train_index], Y.iloc[test_index]
        model.fit(x_tr.to_numpy(), y_tr.to_numpy())
        fpr, tpr, thresholds = roc_curve(y_tr, model.predict(x_tr.to_numpy()))
        roc_auc = auc(fpr, tpr)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        cv_score.append(model.score(x_tr.to_numpy(), y_tr.to_numpy()))
        print(cv_score[-1])
        i += 1
    print('Average K-Fold Score :', np.mean(cv_score))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic for Cross Validation")
    ax.legend(loc="lower right")
    plt.grid()
    plt.savefig("ROC.png")
    plt.show()


if __name__ == "__main__":
    # Reading test and train Data and separate target column
    train_data = pd.read_csv("breast-cancer/train.csv")
    train_y = train_data["diagnosis"].map({'B': 1, 'M': 0}).astype(int)
    train_data = train_data.drop(["diagnosis", 'id'], axis=1)
    test_data = pd.read_csv("breast-cancer/valid.csv")
    test_y = test_data["diagnosis"].map({'B': 1, 'M': 0}).astype(int)
    test_data = test_data.drop(["diagnosis", 'id'], axis=1)

    """
    a) use the dataset as it is.
    """
    first_model = logistic_regression()
    kf = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
    # Cross Validation
    k_fold_validation(kf, first_model, train_data, train_y)
    # train full model
    first_model.fit(train_data.to_numpy(), train_y.to_numpy())
    print("Score on Test Data :", first_model.score(test_data.to_numpy(), test_y.to_numpy()))

    """
     b) Apply z-score normalization
    """
    normalization_z_score = Normalization()
    normalization_z_score.fit(train_data)
    train_normalize_Z = normalization_z_score.transform(train_data)
    test_data_normalize_Z = normalization_z_score.transform(test_data)

    second_model = logistic_regression()

    # Cross Validation
    k_fold_validation(kf, second_model, train_normalize_Z, train_y)
    # train full model
    second_model.fit(train_normalize_Z.to_numpy(), train_y.to_numpy())
    print("Score on Test Data :", second_model.score(test_data_normalize_Z.to_numpy(), test_y.to_numpy()))

    """
    c) Apply zero mean normalization and PCA
    """
    normalization_zero_mean = Normalization(normalization="Zero_mean")
    normalization_zero_mean.fit(train_data)
    train_normalize_Zero = normalization_zero_mean.transform(train_data)
    test_normalize_Zero = normalization_zero_mean.transform(test_data)

    pca_instance = PCA()
    pca_instance.fit(train_normalize_Zero)
    x_train_reduce = pca_instance.transform(train_normalize_Zero)
    x_test_reduce = pca_instance.transform(test_normalize_Zero)

    third_model = logistic_regression()
    # Cross Validation
    k_fold_validation(kf, third_model, x_train_reduce, train_y)
    # train model
    third_model.fit(x_train_reduce.to_numpy(), train_y.to_numpy())
    print("Score on Test Data :", third_model.score(x_test_reduce.to_numpy(), test_y.to_numpy()))

    """
    d) Apply Z_score normalization and PCA
    """

    pca_instance = PCA()
    pca_instance.fit(train_normalize_Z)
    x_train_reduce_Z = pca_instance.transform(train_normalize_Z)
    x_test_reduce_Z = pca_instance.transform(test_data_normalize_Z)

    fourth_model = logistic_regression()

    # Cross Validation
    k_fold_validation(kf, fourth_model, x_train_reduce_Z, train_y)
    # train full model
    fourth_model.fit(x_train_reduce_Z.to_numpy(), train_y.to_numpy())
    print("Score on Test Data :", fourth_model.score(x_test_reduce_Z.to_numpy(), test_y.to_numpy()))