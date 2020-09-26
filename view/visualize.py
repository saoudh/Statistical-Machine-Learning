import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


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


def plot_lda_classification(data,y_true,y_pred,y_pred2):
    nrows=2
    ncols=2
    fig, ax=plt.subplots(nrows=nrows,ncols=ncols)
    plt.subplots_adjust(hspace=1.0)

    label_dict = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3'}
    colors=["red","green","blue","yellow","black","gray"]
    labels=["true data","predicted data","p2"]

    # plot 2 subplots with true labels and predictions
    for i,(y,title) in enumerate(zip([y_true,y_pred,y_pred2],labels)):
        for label,color in zip(range(0,3),colors):
            # compute current row and column index of the subplot
            row_idx = int(abs(i / ncols))
            col_idx = i % 2
            ax[row_idx,col_idx].scatter(x=data[:, 0].real[y == label],
                        y=data[:, 1].real[y == label],
                        color=color,
                        alpha=0.5,
                        label=label_dict[label]
                        )
            # hide grid lines
            ax[row_idx,col_idx].grid(b=False)
            ax[row_idx,col_idx].set_title(title, fontsize="medium")
            ax[row_idx,col_idx].set_xlabel("x")
            ax[row_idx,col_idx].set_ylabel("y")
            ax[row_idx,col_idx].set_aspect('equal', adjustable='box')

    # compute index of last row and column to delete it
    row_idx =1# int(abs(len(n_data) / ncols))
    col_idx = 1# len(n_data) % 2
    # delete last odd axis to put the legend there
    fig.delaxes(ax[row_idx, col_idx])

    # apply legend and plot
    leg=plt.legend(loc="best", bbox_to_anchor=(1.35, 1.2))

    #leg = plt.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    fig.suptitle('LDA-Classification with 3 classes')

    plt.show()


def plot_pca_variance_covering(percent_lst):
    from matplotlib.ticker import MaxNLocator
    fig,ax=plt.subplots(1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(percent_lst)
    ax.set_xlabel("#Dimensions")
    ax.set_ylabel("% variance covered")
    plt.tight_layout()
    plt.show()

def plot_pca_projection(low_dim_data,Y):
    fig, ax = plt.subplots(1)
    label_dict = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3'}
    colors=["red","green","blue","yellow","black","gray"]

    for label,color in zip(range(0,3),colors):
        y_axis=low_dim_data[:,1] if len(low_dim_data.shape)>1 else None
        ax.scatter(x=low_dim_data[:, 0].real[Y == label],
                    y=y_axis.real[Y == label],
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )
        # hide grid lines
        ax.grid(b=False)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_aspect('equal', adjustable='box')


    fig.suptitle('PCA-Classification with 3 classes')
    plt.tight_layout()

    plt.show()

def plot_pca_projection_old(orig_data,low_dim_data,Y):
    ncols=2
    fig, ax=plt.subplots(ncols=ncols)#,ncols=ncols)
    plt.subplots_adjust(hspace=1.0)

    label_dict = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3'}
    colors=["red","green","blue","yellow","black","gray"]
    labels=["orig data","projected data"]

    # plot 2 subplots with true labels and predictions
    for i,(data,title) in enumerate(zip([orig_data,low_dim_data],labels)):
        for label,color in zip(range(0,3),colors):
            # compute current row and column index of the subplot
            row_idx = int(abs(i / ncols))
            col_idx = i % 2
            y_axis=data[:,1] if len(data.shape)>1 else None
            ax[col_idx].scatter(x=data[:, 0].real[Y == label],
                        y=y_axis.real[Y == label],
                        color=color,
                        alpha=0.5,
                        label=label_dict[label]
                        )
            # hide grid lines
            ax[col_idx].grid(b=False)
            ax[col_idx].set_title(title, fontsize="medium")
            plt.xlabel("x")
            plt.ylabel("y")
            ax[col_idx].set_aspect('equal', adjustable='box')

    # compute index of last row and column to delete it
    #row_idx = int(abs(len(n_data) / ncols))
    #col_idx =  len(n_data) % 2
    # delete last odd axis to put the legend there
    #fig.delaxes(ax[row_idx, col_idx])

    # apply legend and plot
    leg=plt.legend(loc="best", bbox_to_anchor=(1.35, 1.2))

    leg = plt.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    fig.suptitle('PCA-Classification with 3 classes')

    plt.show()