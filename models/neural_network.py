import utils.utils as utils
import numpy as np

class NN():
    def __init__(self,input_size,output_size):
        layer1_size = 128
        # number of different classes for softmax
        self.output_size =output_size
        self.input_size=input_size

        # init parameters for hidden layer
        self.W_hidden = utils.init_weights_by_He(self.input_size, layer1_size)
        self.b_hidden = np.zeros(shape=(layer1_size))

        # init parameters for output layer
        self.W_output = utils.init_weights_by_He(layer1_size, self.output_size)
        self.b_out = np.zeros(shape=(layer1_size))

    def relu(self,input:np.ndarray):
        input[input < 0] = 0
        return input

    def softmax(self,input):
        # to avoid NaN we should add a constant to input-exponent
        constant=np.max(input)
        return -np.exp(input-constant) / np.sum(np.exp(input), axis=0)

    def cross_entropy(self, p,q):
        '''
        computing cross entropy loss
        handles only vectors and not matrices
        L= - (y1*ln(y1_pred))+ (y2*ln(y2_pred))+...
        :param p: true distribution
        :param q: estimation of the prediction probability
        :return:
        '''
        loss=-np.sum(np.dot(p,np.log(q)))
        return loss

    def cross_entropy_deriv(self,p, q):
        '''
        derive cross-entropy loss for softmax input
        :param p: true distribution
        :param q: estimation of the prediction probability
        :return:
        '''
        return q-p



    def forward(self,input:np.ndarray):
        self.input=input
        self.a0=np.dot(self.W_hidden,input)+self.b_hidden
        self.z0=self.relu(self.a0)
        self.a1=np.dot(self.W_output,self.z0)+self.b_out
        self.z1=self.softmax(self.a1)
        return self.z1

    def gradient_descent(self,W,):
        pass

    def softmax_deriv(self,S):
        '''

        :param S: Softmax values (1d vector)
        :return:
        '''
        np.diag(S)-S*S.T

    def relu_deriv(self,input):
        '''
        if input>0 then 1 otherwise 0. For input==0 the derivation is undefined but is set to 0 here
        :param input:
        :return:
        '''
        input[input>0]=1
        input[input<=0]=0
        return input


    def backward(self,p,q):
        # update output layer
        lr=0.01
        # compute cross-entropy-error with its derivation
        cross_entropy_deriv=self.cross_entropy_deriv(p,q)

        S=self.z1
        # compute softmax derivative
        softmax_deriv=self.softmax_deriv(S)
        # gradients for bias is cross-entropy-deriv. * softmax-deriv.
        # dL/da1
        gradients_delta_output= cross_entropy_deriv*softmax_deriv
        # gradients for W is additionally multiplied with hidden output z
        # dl/dw1=dL/da1 * da1/dw1
        weight_gradients_hidden_delta=self.z1 *gradients_delta_output
        # w1=w1+lr*dL/dw1
        self.W_output=self.W_output+lr*weight_gradients_hidden_delta
        # b1 = b1 + lr * dL / da1
        self.b_out = self.b_out + lr*gradients_delta_output.reshape(self.b_out.shape)

        # update hidden layer: dL/dw0=dL/da0 * dL/dw0
        # dL/da0=dL/dz0*dz0/da0
        # dL/dz0
        dz0=np.dot(gradients_delta_output,self.W_output)
        relu_deriv=self.relu_deriv(self.a0)
        gradients_delta_input=np.dot(dz0,relu_deriv)
        weight_gradients_input_delta=self.input*gradients_delta_input

        # w0=w0+lr*dL/dw0
        self.W_hidden = self.W_hidden + lr * weight_gradients_input_delta
        # b0 = b0 + lr * dL / da0
        self.b_hidden = self.b_hidden + lr * gradients_delta_input.reshape(self.b_hidden.shape)


