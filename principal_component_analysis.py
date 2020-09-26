import numpy as np
import view.visualize as v
import utils.utils as utils

path_data="data/part2/iris.txt"

data=utils.read_data_from_txt(path_data,dim=(-1,5),delimiter=",")

# extract feature data
X=data[:,:4]
# extract labels
Y=data[:,4]

def normalize_data(X):
    """
    To normalize results, computing centered mean of input data with unit variance by subtracting X from its mean across the features
    and dividing by standard deviation of X

    :param X: data input
    :return: normalized data
    """
    # centering
    X_center = X - np.mean(X, axis=0)
    # scaling
    X_scaled= X_center / np.std(X, axis=0)

    return X_scaled


def compute_cov(X):
    return np.cov(X.T)


X_norm=normalize_data(X)

# computing covariance matrix of normalized data
cov=compute_cov(X_norm)

# computing eigenvectors/eigenvalues
eig_vals,eig_vecs=np.linalg.eig(cov)
# create pairs of eigenvectors/eigenvalues to sort them pairwise
eig_pairs=[(np.abs(eig_val),eig_vec.tolist()) for eig_val, eig_vec in zip(eig_vals,eig_vecs)]
# sort eigenvalues and corresponding eigenvectors according to the eigenvalue from large to small value
eig_pairs_sorted=sorted(eig_pairs,key=lambda x: x[0],reverse=True)

# compute percentage
eig_pairs_percent=[sum(eig_vals[:i+1])/sum(eig_vals)*100 for i,(pair1,pair2) in enumerate(eig_pairs_sorted)]
# plot the relation of number of dimensions and the accumulated covered variance in percentage
v.plot_pca_variance_covering([0]+eig_pairs_percent)

# compute min number of dimensions necessary to explain 95% of the data
# first compute all indexes of eigenvalues till 0.95
eig_pairs_min=[i for i,(pair1,pair2) in enumerate(eig_pairs_sorted) if (sum(eig_vals[:i+1])/sum(eig_vals))<0.95]
# then increment the index, so we explain more than 0.95
eig_pairs_min+=[eig_pairs_min[-1]+1] if len(eig_pairs_min)>0 else [0]


# assign all eigenvalue/eigenvector pairs
eig_pairs_reduced=np.array(eig_pairs_sorted)[eig_pairs_min]

# create the weight-vector consisting the eigenvectors
W=np.array(eig_pairs_reduced[:,1:].tolist()).reshape((-1,len(eig_pairs_min)))

# create the projected low-dimensional space
X_low=np.dot(X_norm, W)

# plot low dimensional PCA space
v.plot_pca_projection(X_low,Y)

# compute NRMSE for different numbers of principal components
for i in range(1, 5):
    # computing the weight vector with different number i of Principal Components
    eig_pairs_reduced = np.array(eig_pairs_sorted)[:i]
    W = np.array(eig_pairs_reduced[:, 1:].tolist()).reshape((4, i))

    # project to low dimensional space
    X_low = np.dot(X_norm, W)

    # reconstruct to original (normalized) space
    X_reconstr = np.dot(X_low, W.T)

    # computing NRMSE
    nrmse = np.sqrt(np.mean(X_norm - X_reconstr, 0) ** 2 / np.mean(X, 0) ** 2)
    print("i=", i, " - nrmse=", nrmse)
