import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from utils import gaussian_density,gaussian_density_optim
from utils import read_data_from_txt


path_c1="data/part1/densEst1.txt"
path_c2="data/part1/densEst2.txt"
data_c1=read_data_from_txt(path_c1,dim=(-1,2))
data_c2=read_data_from_txt(path_c2,dim=(-1,2))

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





def plot_contour(sigma,mu,prior):
    def g1(x, y):
        probs=np.log(prior[0])+compute_log_likelihood_optim(X=np.column_stack((x, y)), mu=mu[0], sigma=sigma[0])
        return np.array(probs)

    def g2(x, y):
        probs=np.log(prior[1])+compute_log_likelihood_optim(X=np.column_stack((x, y)), mu=mu[1], sigma=sigma[1])
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
    p = (g1(x, y) - g2(x, y)).reshape(X.shape)

    #scatter class 1
    ax0.scatter(data_c1[:,0], data_c1[:, 1],c="red")
    # scatter class 2
    ax0.scatter(data_c2[:, 0], data_c2[:, 1],c="green")

    ax0.contour(X, Y, p, levels=[0])

    plt.show()


def compute_prior(Y):
    '''

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
plot_contour(sigma,mu,prior=prior)

# plot the 3D surface for each class
plot_surface(sigma[0],mu[0])
plot_surface(sigma[1],mu[1])

############### evaluation ##################
# compute the accuracy of the predictions
y_pred = predict(X, prior, mu, sigma)
print("Accuracy: ",np.mean(Y==y_pred))
