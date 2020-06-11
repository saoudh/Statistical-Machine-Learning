import numpy as np


def read_data_from_txt(path, dim):
    '''
    reading data of indicated dimension

    :param path: path to txt-file
    :param dim: dimension of the matrix in the txt-file
    :return: numpy array containing all data points
    '''
    with open(path, "r") as f:
        data = f.read()

    return np.array(data.split(), dtype=np.float32).reshape(dim)

def gaussian_density(x, mu, sigma):
    '''
    computing the gaussian probability of the passed data point x

    :param x: single data point
    :param mu: mean across each dimension
    :param sigma: covariance matrix
    :return: probability of the passed data point x
    '''
    # dimension of x
    d =sigma.shape[0]

    # defining last part of the equation
    eq1 = np.dot((x - mu).T, np.linalg.inv(sigma))
    eq = np.dot(eq1, (x - mu))

    p = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(sigma))) * np.exp(-0.5 * eq)
    if np.isnan(p) or np.isinf(p):
        print("p is nan")
    return p

def gaussian_density_batch(X, mu, sigma):
    '''
    same as function gaussian_density(.), but a batch of data points instead of a single data point is processed at once

    :param X: batch of data points
    :param mu: mean across each dimension
    :param sigma: covariance matrix
    :return: numpy array of the probabilities of each data point in the batch X
    '''

    # dimension of x
    d = sigma.shape[0]

    # defining last part of the equation
    # using tensordot-function instead of dot to process a batch of data points
    eq1 = np.tensordot((X - mu), np.linalg.inv(sigma), 1)
    eq = np.diag(np.tensordot(eq1, (X - mu), [1, 1]))

    p = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(sigma))) * np.exp(-0.5 * eq)

    return p


def gaussian_density_optim(X, mu, sigma):
    '''
    optimized equation of the function gaussian_density(.) by making use of the independendy assumption of Naive Bayes

    :param X: batch of data points
    :param mu: mean across each dimension
    :param sigma: covariance matrix
    :return: numpy array of the probabilities of each data point in the batch X
    '''

    # dimension of x
    d = sigma.shape[0]

    # defining last part of the equation
    eq=np.exp(-0.5 * np.sum((X - mu) ** 2 / np.diag(sigma), axis=1))

    p = 1 / ((2 * np.pi) ** (d / 2) * np.prod(np.sqrt(np.diag(sigma)))) *eq

    return p
