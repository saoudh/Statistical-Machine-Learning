import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from utils.utils import gaussian_density_optim


def plot_rmse(X,fig_title,fig_subtitle,**args):
    colors = ["red", "green", "blue", "yellow", "black", "gray"]
    if type(list(args.values())[0]) is dict:
        args = list(args.values())[0]
    for (k, v), c in zip(args.items(), colors):
        plt.plot(X, v, c=c, label=k)
    plt.suptitle(fig_subtitle)
    plt.title(fig_title)
    plt.xlabel("polynom degree")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

def plot_multiple_curves(X,fig_title,fig_subtitle,**args):
    colors=["red","green","blue","yellow","black","gray"]
    if type(list(args.values())[0]) is dict:
        args=list(args.values())[0]
    for (k,v),c in zip(args.items(),colors):
        plt.scatter(X, v, c=c, label=k)
    plt.suptitle(fig_subtitle)
    plt.title(fig_title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()



def create_subplot_predictive_distributions_uncertainty(data_true_full, data_true_partly, data_pred, std_err, fig_subtitle, ax):
    colors=["red","green","blue","yellow","black","gray"]
    data_true_full=np.array(sorted(data_true_full,key=lambda x: x[0]))
    #for (k,tup),c in zip(data_true_full,colors):
    ax.plot(data_true_full[:,0], data_true_full[:,1], c=colors[0], label="true data")
    ax.scatter(data_true_partly[0,:], data_true_partly[1,:], c=colors[1],label="partly data")
    ax.plot(data_pred[0,:], data_pred[1,:], c=colors[2], label="pred data")
    ax.fill_between(data_pred[0,:], data_pred[1,:]-std_err,data_pred[1,:]+std_err,label="pred data")
    ax.set_title(fig_subtitle,fontsize="medium")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
