import numpy as np
import view.visualize as v
import utils.utils as utils

path_data="data/part2/ldaData.txt"

data=utils.read_data_from_txt(path_data,dim=(-1,2))

# specify the sample size of each class
c1_len=50
c2_len=43
c3_len=44

# data is divided into 3 classes
c1=data[:c1_len]
c2=data[c1_len:c1_len+c2_len]
c3=data[c1_len+c2_len:]

# create label vector with classes 0, 1 and 2
y_true=np.hstack([np.zeros([c1_len]), 1 * np.ones([c2_len]), 2 * np.ones([c3_len])]).ravel()

def compute_mean_vec(data):
    '''
    compute mean for each feature of the data for a class

    :param data:
    :return:
    '''
    return np.mean(data,axis=0)

def compute_Sb_between_class_cov(data,**args):
    '''
    Computing the within class scatter matrix

    :param data: original data
    :param args: data of each class
    :return:
    '''
    m=compute_mean_vec(data)
    s = np.array(list(map(lambda c: len(c) * np.outer((compute_mean_vec(c) - m),
                                                  (compute_mean_vec(c) - m)),
                        args.values())))
    s=np.sum(s,axis=0)

    return s

def compute_Sw_between_class_cov(**args):
    '''
    Computing the between class scatter matrix

    :param args: data of each class
    :return:
    '''
    c=np.array(list(map(lambda c: np.sum([np.outer((x - compute_mean_vec(c)),
                    (x - compute_mean_vec(c))) for x in c], axis=0),args.values())))

    s=np.sum(c,axis=0)

    return s

# compute between class matrix
Sb=compute_Sb_between_class_cov(data=data,c1=c1,c2=c2,c3=c3)

# compute within class matrix
Sw=compute_Sw_between_class_cov(c1=c1,c2=c2,c3=c3)

# compute all Eigenvectors of (Sw^-1 * Sb)
_,eigenvecs=np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

# build matrix W consisting of each Eigenvector row wise
W = np.stack([eigenvecs[0],eigenvecs[1]],axis=0)

# compute the projected data by multiplying original 2-dim data with computed matrix W
data_lda = data.dot(W)

# compute bayesian posterior p(c|x)=p(c)p(x|c) for each class c using projected data and stack them together R^(N,dim)
probs=np.stack([len(data_lda[:c1_len]) / len(data_lda) * utils.gaussian_density_optim(data_lda, mu=np.mean(data_lda[:c1_len], axis=0), sigma=np.cov(data_lda[:c1_len], rowvar=False)),
                len(data_lda[c1_len:c1_len + c2_len]) / len(data_lda) * utils.gaussian_density_optim(data_lda, mu=np.mean(data_lda[c1_len:c1_len + c2_len], axis=0), sigma=np.cov(data_lda[c1_len:c1_len + c2_len], rowvar=False)),
                len(data_lda[c1_len + c2_len:]) / len(data_lda) * utils.gaussian_density_optim(data_lda, mu=np.mean(data_lda[c1_len + c2_len:], axis=0), sigma=np.cov(data_lda[c1_len + c2_len:], rowvar=False))], axis=1)

# compute bayesian posterior p(c|x)=p(c)p(x|c) for each class c using projected data and stack them together R^(N,dim)
probs2=np.stack([len(data_lda[:c1_len]) / len(data_lda) * utils.gaussian_density_optim(data, mu=np.mean(data[:c1_len], axis=0), sigma=np.cov(data[:c1_len], rowvar=False)),
                len(data_lda[c1_len:c1_len + c2_len]) / len(data_lda) * utils.gaussian_density_optim(data, mu=np.mean(data[c1_len:c1_len + c2_len], axis=0), sigma=np.cov(data[c1_len:c1_len + c2_len], rowvar=False)),
                len(data_lda[c1_len + c2_len:]) / len(data_lda) * utils.gaussian_density_optim(data, mu=np.mean(data[c1_len + c2_len:], axis=0), sigma=np.cov(data[c1_len + c2_len:], rowvar=False))], axis=1)


# predict the class for each sample by applying argmax
y_pred=np.argmax(probs,axis=1)
y_pred2=np.argmax(probs2,axis=1)

print("Accuracy:",np.sum(y_true==y_pred)/len(y_pred))
print("True predictions:",np.sum(y_true==y_pred), " out of ", len(y_true)," samples.")
print("Missclasified samples:",len(y_true)-np.sum(y_true==y_pred), " out of ", len(y_true)," samples.")
print("Missclasified samples 2:",len(y_true)-np.sum(y_true==y_pred2), " out of ", len(y_true)," samples.")


# plot
v.plot_lda_classification(data=data, y_true=y_true, y_pred=y_pred,y_pred2=y_pred2)