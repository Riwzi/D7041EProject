"""
This code is adopted under free software license for educational purposes
"""

import numpy as np
import sys

def f_sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return f_sigmoid(X)*(1 - f_sigmoid(X))

def f_tanh(X, deriv=False):
    if not deriv:
        return np.tanh(X)
    else:
        return 1.0 - f_tanh(X)**2

def f_staircase(X, deriv=False):
    a = 100
    N = 4
    k = 3
    j = np.arange(1.,N)
    # the third dimension is used for different values of j and is then summed
    jN = (j/N).reshape(1,1,N-1)
    # j/N is repeated to fit the shape of X
    JN = np.repeat(np.repeat(jN, X.shape[0], axis=0), X.shape[1], axis=1)
    # X is repeated to fit the third dimension of JN
    X = np.repeat(X.reshape(X.shape[0], X.shape[1], 1), JN.shape[2], axis=2)
    if not deriv:
        # 1/2 + 1/(2(k-1))*sum(tanh(a(X - j/N)), j=1 to N-1)
        return .5 + np.sum(np.tanh(a * (X - JN)), axis=2)/(2*k-2)
    else:
        # 1/(2(k-1))sum(a*sech(a*(x-j/n))**2, j=1 to N-1)
        # sech = 1/cosh
        return np.sum(a/np.cosh(a * (X - JN)))/(2*k-2)

def f_softmax(X):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z

def f_linear(X):
    return X

def exit_with_err(self, err_str):
    print >> sys.stderr, err_str
    sys.exit(1)



class Layer:
    def __init__(self, size, batch_size, is_input=False, is_output=False,
                 activation=f_sigmoid):
        self.is_input = is_input
        self.is_output = is_output

        # Z is the matrix that holds output values
        self.Z = np.zeros((batch_size, size[0]))
        # The activation function is an externally defined function (with a
        # derivative) that is stored here
        self.activation = activation

        # W is the outgoing weight matrix for this layer
        self.W = None
        # S is the matrix that holds the inputs to this layer
        self.S = None
        # D is the matrix that holds the deltas for this layer
        self.D = None
        # Fp is the matrix that holds the derivatives of the activation function
        self.Fp = None

        if not is_input:
            self.S = np.zeros((batch_size, size[0]))
            self.D = np.zeros((batch_size, size[0]))

        if not is_output:
            self.W = np.random.normal(size=size, scale=1E-4)

        if not is_input and not is_output:
            self.Fp = np.zeros((size[0], batch_size))

    def forward_propagate(self):
        if self.is_input:
            return self.Z.dot(self.W)

        self.Z = self.activation(self.S)
        if self.is_output:
            return self.Z
        else:
            # For hidden layers, we add the bias values here
            self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
            self.Fp = self.activation(self.S, deriv=True).T
            return self.Z.dot(self.W)


class MultiLayerPerceptron(object):
    def __init__(self, layer_config, batch_size=100):
        self.layers = []
        self.num_layers = len(layer_config)
        self.minibatch_size = batch_size

        for i in range(self.num_layers-1):
            if i == 0:
                print "Initializing input layer with size {0}.".format(
                    layer_config[i]
                )
                # Here, we add an additional unit at the input for the bias
                # weight.
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                         batch_size,
                                         is_input=True))
            else:
                print "Initializing hidden layer with size {0}.".format(
                    layer_config[i]
                )
                # Here we add an additional unit in the hidden layers for the
                # bias weight.
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                         batch_size,
                                         activation=f_sigmoid))

        print "Initializing output layer with size {0}.".format(
            layer_config[-1]
        )
        self.layers.append(Layer([layer_config[-1], None],
                                 batch_size,
                                 is_output=True,
                                 activation=f_softmax))
        print "Done!"

    def forward_propagate(self, data):
        # We need to be sure to add bias values to the input
        self.layers[0].Z = np.append(data, np.ones((data.shape[0], 1)), axis=1)

        for i in range(self.num_layers-1):
            self.layers[i+1].S = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, yhat, labels):
        self.layers[-1].D = (yhat - labels).T
        for i in range(self.num_layers-2, 0, -1):
            # We do not calculate deltas for the bias values
            W_nobias = self.layers[i].W[0:-1, :]
            self.layers[i].D = W_nobias.dot(self.layers[i+1].D) * self.layers[i].Fp

    def update_weights(self, eta):
        for i in range(0, self.num_layers-1):
            W_grad = -eta*(self.layers[i+1].D.dot(self.layers[i].Z)).T
            self.layers[i].W += W_grad

    def evaluate(self, train_data, train_labels, test_data, test_labels,
                 num_epochs=70, eta=0.05, eval_train=False, eval_test=True):

        N_train = len(train_labels)*len(train_labels[0])
        N_test = len(test_labels)*len(test_labels[0])

        print "Training for {0} epochs...".format(num_epochs)
        for t in range(0, num_epochs):
            out_str = "[{0:4d}] ".format(t)

            for b_data, b_labels in zip(train_data, train_labels):
                output = self.forward_propagate(b_data)
                self.backpropagate(output, b_labels)
                self.update_weights(eta=eta)

            if eval_train:
                errs = 0
                for b_data, b_labels in zip(train_data, train_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = "{0} Training error: {1:.5f}".format(out_str,
                                                           float(errs)/N_train)

            if eval_test:
                errs = 0
                for b_data, b_labels in zip(test_data, test_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = "{0} Test error: {1:.5f}".format(out_str,
                                                       float(errs)/N_test)

            print out_str

class ReplicatorNeuralNet(MultiLayerPerceptron):
    def __init__(self, layer_config, batch_size=100):
        assert(len(layer_config) == 5)
        assert(layer_config[0] == layer_config[-1])
        super(ReplicatorNeuralNet, self).__init__(layer_config, batch_size)
        self.layers[1].activation = f_tanh
        self.layers[2].activation = f_staircase
        self.layers[3].activation = f_tanh
        # article uses either linear or sigmoid for output layer
        self.layers[4].activation = f_linear

    def train(self, train_data, epochs, learning_rate):
        loss = np.array([])
        for i in range(1,epochs+1):
            epoch_loss = np.array([])
            for batch_data in train_data:
                batch_loss = train_once(batch_data)
                # store mean loss for each epoch so we can plot it
                epoch_loss = np.append(epoch_loss, batch_loss)
                # TODO log it
            loss = np.append(loss, np.mean(epoch_loss))
        return losses

    def train_once(self, train_data, learning_rate):
        output = self.forward_propagate(train_data)
        # the output is compared to the INPUT for training the RNN
        self.backpropagate(output, train_data)
        self.update_weights(learning_rate)
        # mean loss over the whole batch
        return np.mean(self.reconstruction_loss(train_data, output), axis=0)

    def reconstruction_loss(self, net_input, net_output):
        # half square euclidean distance is used for reconstruction loss
        np.sum(np.square(net_output - net_input), axis=1)/2

    def evaluate_outlier_thresholds(self, data, labels):
        output = self.forward_propagate(data)
        outlier_factor = self.reconstruction_loss(data, output)
        assert(len(outlier_factor.shape) == 1)
        # sort and reverse so that first index is that of the element with
        # biggest outlier factor
        outlier_indices = np.argsort(outlier_factor)[::-1]
        # correct user
        positive_examples = np.sum(labels)
        # incorrect user
        negative_examples = positive_examples - labels.shape[0]
        fa = negative_examples # number of false acceptances
        fr = 0                 # number of false rejections
        threshold = []
        far = []
        frr = []
        for i in outlier_indices:
            # label is 1 if it is the original user, 0 otherwise
            if labels[i] == 1:
                # lowering the threshold this far will reject
                fr += 1
            else:
                fa -= 1
            threshold.append(outlier_factor[i])
            far.append(fa / float(negative_examples))
            frr.append(fr / float(positive_examples))
        return (threshold, far, frr)

    def find_outliers(self, net_input, net_output=None):
        if net_output == None:
            net_output = self.forward_propagate(net_input)
        outlier_factor = self.reconstruction_loss(net_input, net_output)
        outliers = np.where(outlier_factor > self.outlier_threshold, 1, 0)
        return outliers
