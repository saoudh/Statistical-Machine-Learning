import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from utils.utils import gaussian_density_optim


def plot_surface(sigma,mu):
    '''
    plotting 3D surface of the gaussian distributions with the passed parameter

    :param sigma: covariance matrix
    :param mu: array with mean of each dimension
    :return:
    '''

    # Creating 2-dim meshgrid for pos -10 to 10 with 500 steps for each dim
    steps=500
    x = np.linspace(-10, 10, steps)
    y = np.linspace(-10, 10, steps)
    X, Y = np.meshgrid(x, y)

    # init meshgrid with dim (500,500,2) to create pairs of each possible position in the 2-dim grid
    grid = np.empty(X.shape + (2,))

    # assign the positions for X at first pair position
    grid[:, :, 0] = X
    # assign the positions for X at second pair position
    grid[:, :, 1] = Y

    # compute gaussian probabilities for a 3D environment: 500x500x2 dimension
    pdf = np.array([gaussian_density_optim(grid[i],mu=mu,sigma=sigma) for i in range(grid.shape[0])])

    # draw the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, pdf, cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def plot_contour(sigma,mu,prior,data_c1,data_c2):
    '''
    Plotting the contour and the decision boundary of both classes

    :param sigma: covariance matrix
    :param mu: array with mean of each dimension
    :param prior: array with the prior of each class
    :return:
    '''
    import naive_bayes_w_gaussians as GNB

    # log-posterior/classifier
    def g(x, y,idx_class):
        probs=np.log(prior[idx_class])+GNB.compute_log_likelihood_optim(X=np.column_stack((x, y)), mu=mu[idx_class], sigma=sigma[idx_class])
        return np.array(probs)

    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)

    pos = np.array([X.flatten(), Y.flatten()]).T

    rv3=np.array(gaussian_density_optim(pos, mu[0], sigma[0]))
    rv4=np.array(gaussian_density_optim(pos, mu[1], sigma[1]))

    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)

    ax0.contour(X, Y, rv3.reshape(500, 500), cmap='RdGy')
    ax0.contour(X, Y, rv4.reshape(500, 500), 20, cmap='BrBG')
    x=X.flatten()
    y=Y.flatten()
    p = (g(x, y, idx_class=0) - g(x, y, idx_class=1)).reshape(X.shape)

    #scatter class 1
    ax0.scatter(data_c1[:,0], data_c1[:, 1],c="red")
    # scatter class 2
    ax0.scatter(data_c2[:, 0], data_c2[:, 1],c="green")

    ax0.contour(X, Y, p, levels=[0])

    plt.show()
