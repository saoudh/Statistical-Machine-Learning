import numpy as np
import view.visualize as v
import utils.utils as utils
import models.neural_network


path_data_train_in="data/part3/mnist_small_train_in.txt"
path_data_train_out="data/part3/mnist_small_train_out.txt"
path_data_test_in="data/part3/mnist_small_test_in.txt"
path_data_test_out="data/part3/mnist_small_test_out.txt"

data_train_in=utils.read_data_from_txt(path_data_train_in,delimiter=",")
data_test_in=utils.read_data_from_txt(path_data_test_in,delimiter=",")
data_train_out=utils.read_data_from_txt(path_data_train_out,dim=(-1))
data_test_out=utils.read_data_from_txt(path_data_test_out,dim=(-1))
print(np.shape(data_train_out))

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
data_train_out_onehot=enc.fit_transform(data_train_out)
data_test_out_onehot=enc.fit_transform(data_test_out)


output_size=len(np.unique(data_train_out))
input_size=data_train_in.shape[1]
NN=models.neural_network.NN(input_size=input_size,output_size=output_size)

mini_batch_size=32

train_len=int(np.ceil(data_train_in.shape[0]/mini_batch_size))
test_len=int(np.ceil(data_test_in.shape[0]/mini_batch_size))

def train():
    for i in range(train_len):
        mini_batch_in=data_train_in[i * mini_batch_size:mini_batch_size * (i + 1)]
        mini_batch_out=data_train_out[i * mini_batch_size:mini_batch_size * (i + 1)]
        prob_output=NN.forward(mini_batch_in)
        NN.backward(p=prob_output,q=mini_batch_out)

# testing is without backpropagation step to not modify the weights
def test():
    for i in range(test_len):
        mini_batch_in=data_test_in[i * mini_batch_size:mini_batch_size * (i + 1)]
        mini_batch_out=data_test_out[i * mini_batch_size:mini_batch_size * (i + 1)]
        prob_output=NN.forward(mini_batch_in)
        # todo: check whether prob=labels

epochs=3
for i in range(epochs):
    train()
    test()
