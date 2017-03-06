'''
originally by eao
'''
import numpy as np
import matplotlib.pyplot as plt
from NearestNeighbor import *
from Backpropagation import *

ds=DataSet()
util=Utilities()

batch_size=100;

batched_data, batched_labels = util.create_batches(data, labels, batch_size,
                                                   create_bit_vector=True)
rnn = ReplicatorNeuralNet(layer_config=[31, 15, 15, 15, 31], batch_size=batch_size)
loss = rnn.train(train_data)

plt.plot(loss)
plt.xlabel("epoch")
plt.ylabel("reconstruction loss")
plt.title("Replicator neural network training")
plt.show()

# TODO use more data for evaluation?
thresholds, far, frr = rnn.evaluate_outlier_thresholds(data, labels)
# TODO make the plot nicer
plt.plot(thresholds, far, 'x')
plt.plot(thresholds, frr, '.')
plt.xlabel("threshold")
plt.ylabel("rate")
plt.title("Replicator neural network FAR/FRR")
plt.show()
