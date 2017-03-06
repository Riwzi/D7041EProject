'''
originally by eao
'''
import numpy as np
import matplotlib.pyplot as plt
from NearestNeighbor import *
from Backpropagation import *

print("Hello World:)\n")
ds=DataSet()
util=Utilities()

util.test()

Train, Valid, Test = ds.load_MNIST()
Train_images=Train[0]
Train_labels=Train[1]
Valid_images=Valid[0]
Valid_labels=Valid[1]

def prepare_for_backprop(batch_size, Train_images, Train_labels, Valid_images, Valid_labels):

    print "Creating data..."
    batched_train_data, batched_train_labels = util.create_batches(Train_images, Train_labels,
                                              batch_size,
                                              create_bit_vector=True)
    batched_valid_data, batched_valid_labels = util.create_batches(Valid_images, Valid_labels,
                                              batch_size,
                                              create_bit_vector=True)
    print "Done!"


    return batched_train_data, batched_train_labels,  batched_valid_data, batched_valid_labels

batch_size=100;

train_data, train_labels, valid_data, valid_labels=prepare_for_backprop(batch_size, Train_images, Train_labels, Valid_images, Valid_labels)

rnn = ReplicatorNeuralNet(layer_config=[31, 15, 15, 15, 31], batch_size=batch_size)
loss = rnn.train(train_data)

plt.plot(loss)
plt.xlabel("epoch")
plt.ylabel("reconstruction loss")
plt.title("Replicator neural network training")
plt.show()
# mlp.evaluate(train_data, train_labels, valid_data, valid_labels,
            #  eval_train=True)
