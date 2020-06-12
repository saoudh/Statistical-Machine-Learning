import numpy as np
from utils.utils import read_data_from_txt
import view.visualize_gaussians

def compute_mle_mu(X:np.ndarray):
    '''
    computing the mean across each dimension

    :param X: multidimensional data as numpy-matrix
    :return: array of the means across each dimension
    '''
    # mean across each dimension
    mu=X.mean(axis=0)

    return mu

def compute_mle_sigma(X:np.ndarray, biased=False):
    '''
    computing the MLE for the covariance

    :param X: multidimensional data as numpy-matrix
    :param biased: Whether to use unbiased function or not
    :return: numpy matrix with MLE for the covariance
    '''
    # mean across each dimension
    mu=X.mean(axis=0)

    # number of data points
    N=X.shape[0]

    # number of dimensions
    cols=X.shape[1]

    # initiate covariance matrix with zeros of size (dim,dim)
    mle_sigma=np.zeros([cols,cols])

    if biased:
        # computing biased MLE for sigma
        for x in range(cols):
            for y in range(cols):
                mle_sigma[x][y]=1/N*np.sum((np.stack(X,1)[x]-mu[x])*(np.stack(X,1)[y]-mu[y]).T,axis=0)
    else:
        # computing unbiased MLE for sigma
        for x in range(cols):
            for y in range(cols):
                mle_sigma[x][y]=1/(N-1)  *np.sum((np.stack(X,1)[x]-mu[x])*(np.stack(X,1)[y]-mu[y]).T,axis=0)

    return mle_sigma

def compute_log_likelihood(x,sigma,mu):
    '''
    slower version of log-likelihood where the indepence of each variable/dimension is not assumed

    :param x: single data point
    :param sigma: covariance matrix
    :param mu: mean across each dimension
    :return: single value with log-likelihood
    '''
    # number of dimensions
    d=sigma.shape[0]

    # defining last part of the equation
    eq1=np.dot((x - mu).T,np.linalg.inv(sigma))
    eq=np.dot(eq1,(x-mu))

    return -np.log((2*np.pi)**(d/2))-np.log(np.sqrt(np.linalg.det(sigma))) - 0.5 * eq

def compute_log_likelihood_optim(X, sigma, mu):
    '''
    optimized version of log-likelihood where the indepence of each variable/dimension is assumed

    :param X: numpy matrix containing all data points
    :param sigma: covariance matrix
    :param mu: mean across each dimension
    :return: numpy array with log-likelihood for each data point in X
    '''
    # number of dimensions
    d=sigma.shape[0]

    # defining last part of the equation
    eq=np.sum((X - mu) ** 2 / np.diag(sigma), axis=1)

    return -np.log((2*np.pi)**(d/2)*np.prod(np.sqrt(np.diag(sigma)))) - 0.5 * eq


def compute_prior(Y):
    '''
    Computing the prior for each class

    :param Y: computing the labels by counting the number of data points for each class
            and dividing it by the sum of all data points
    :return: numpy array with priors for each class
    '''
    classes,counts=np.unique(Y,return_counts=True)
    prior=np.array([x/len(Y) for x in counts])
    return prior



def predict(X,prior,mu, sigma):
    '''

    :param X: numpy matrix with all data points
    :param prior: numpy array with the prior value of each class
    :param mu: means for each class of each dimension
    :param sigma: covariance matrix for each class
    :return: numpy array with values 0 (class 1) or 1 (class 2) for each data point
    '''
    # computing the log-posterior, which is the probability that it belongs to each class
    probs=[np.log(prior[c])+compute_log_likelihood_optim(X,sigma=sigma[c],mu=mu[c]) for c in range(len(prior))]

    # select class with highest probability
    y=np.argmax(probs,axis=0)
    return y

if __name__ =="__main__":
    # reading data from txt-file for class 1 and class 2, which are saved in seperate files
    path_c1="data/part1/densEst1.txt"
    path_c2="data/part1/densEst2.txt"
    data_c1=read_data_from_txt(path_c1,dim=(-1,2))
    data_c2=read_data_from_txt(path_c2,dim=(-1,2))

    ############# init parameters #################
    sigma=[]
    mu=[]

    # compute params for class 1
    sigma.append(compute_mle_sigma(data_c1,biased=False))
    mu.append(compute_mle_mu(data_c1))

    # compute params for class 2
    sigma.append(compute_mle_sigma(data_c2,biased=False))
    mu.append(compute_mle_mu(data_c2))

    # computing the labels: data in data_c1 belongs to class 1 and data in data_c2 to class 2
    Y=np.append(np.zeros([len(data_c1)]),(np.ones([len(data_c2)])))

    # stack data points of both class
    X=np.vstack([data_c1,data_c2])

    # computing the prior for both classes
    prior=compute_prior(Y)

    ############### plotting ######################
    # plot the contour with the decision border for both classes
    view.visualize_gaussians.plot_contour(sigma=sigma,mu=mu,prior=prior,data_c1=data_c1,data_c2=data_c2)

    # plot the 3D surface for each class
    view.visualize_gaussians.plot_surface(sigma=sigma[0],mu=mu[0])
    view.visualize_gaussians.plot_surface(sigma=sigma[1],mu=mu[1])

    ############### evaluation ##################
    # compute the accuracy of the predictions
    y_pred = predict(X, prior, mu, sigma)
    print("Accuracy: ",np.mean(Y==y_pred))
