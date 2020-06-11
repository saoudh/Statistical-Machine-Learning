import numpy as np
def read_data_from_txt(path,dim):
    '''
    reading 2-dim data
    :param path:
    :return:
    '''
    with open(path, "r") as f:
        data= f.read()

    return np.array(data.split(),dtype=np.float32).reshape(dim)

def gaussian_density(x, mu, sigma):
    # p=1/(np.sqrt(2*math.pi)*sigma)*np.exp(-0.5*((x-mu)/sigma)**2)
    # dimension of x
    d = len(x)
    f1 = np.dot((x - mu).T, np.linalg.inv(sigma))
    #print("1.f1=",f1)
    f2 = np.dot(f1, (x - mu))
    #print("1.f2=", f2)
    p = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(sigma))) * np.exp(-0.5 * f2)
    if np.isnan(p) or np.isinf(p):
        print("p is nan")
    return p

def gaussian_density_batch(x, mu, sigma):
    # p=1/(np.sqrt(2*math.pi)*sigma)*np.exp(-0.5*((x-mu)/sigma)**2)
    # dimension of x
    d = 2
    f1 = np.tensordot((x - mu), np.linalg.inv(sigma),1)
    #print("2.f1=",f1)
    f2 = np.diag(np.tensordot(f1, (x - mu),[1,1]))
    #print("2.f2=",f2)
    p = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(sigma))) * np.exp(-0.5 * f2)

    return p


def gaussian_density_optim(x, mu, sigma):
    # dimension of x
    d = x.shape[1]

    f2=np.exp(-0.5 * np.sum((x-mu)**2/np.diag(sigma),axis=1))
    p = 1 / ((2 * np.pi) ** (d / 2) * np.prod(np.sqrt(np.diag(sigma)))) *f2

    return p

#X=np.array([[1.3,1.2],[1.2,1.3],[1.2,1.4]])
X=np.ones([1000,2])
mu=np.array([1.1,1.2])
sigma=np.array([[1.1,0],[0,1.1]])
from time import time

start1 = time()

prob=[gaussian_density(x,mu,sigma) for x in X]
#print("1.",prob)

print("time slower=", time() - start1)
first=time() - start1
#print(prob)
start2 = time()

prob=gaussian_density_batch(X, mu, sigma)
#print("2.",prob)

print("time batch=", time() - start2)
second=time() - start2

print(second/first*100)

start2 = time()

prob=gaussian_density_optim(X, mu, sigma)
print("time optim=", time() - start2)
third=time() - start2

print(second/first*100)
print(third/first*100)

#print("3.",prob)
