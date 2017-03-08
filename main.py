'''
Created on 28 feb. 2017

@author: Rickard
'''
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from nearestneighbor import *
from Backpropagation import *
from cross_validation import cross_validation

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("train_correct", help="Number of correct user keystrokes in the training data", type=int)
    parser.add_argument("train_other", help="Number of other users keystrokes in the training data", type=int)
    parser.add_argument("subject", help="Id string of the correct subject.", type=str)

    # Optional Arguments
    parser.add_argument("-m", help="Method. 0 for KNN, 1 for MLP, 2 for RNN. Defaults to 0", type=int, default="0")

    # Parse arguments
    args = parser.parse_args()

    return args

args=parseArguments()
#data/labels for  the correct subjects and other subjects.
correct_data, correct_labels, other_data, other_labels = [], [], [], []
correct_subject = args.subject

#Read all data of the correct user into a separate array. Easier to split the data this way
filepath = os.path.join('data', 'DSL-StrongPasswordData.txt')
text_file = open(filepath, 'r')
text_file.next() #ignore the first line
for line in text_file:
    x = line.split()
    label = x[0]
    data = map(float, x[3:]) #convert to floats from strings
    if (label==correct_subject):
        correct_data.append(data)
        correct_labels.append(label)
    else:
        other_data.append(data)
        other_labels.append(label)
text_file.close()

#Convert to numpy arrays
correct_data = np.array(correct_data)
correct_labels = np.array(correct_labels)
other_data = np.array(other_data)
other_labels = np.array(other_labels)

#Convert all labels with correct subject to 1, all other labels to 0
correct_labels = np.ones(correct_labels.shape, dtype=np.int)
other_labels = np.zeros(other_labels.shape, dtype=np.int)

def prepare_data(correct_data, correct_labels, other_data, other_labels, n, m):
    #First shuffle the data
    correct_data, correct_labels = shuffle(correct_data, correct_labels)
    other_data, other_labels = shuffle(other_data, other_labels)

    #Pick the first n keystrokes of the correct user and the first m keystrokes of incorrect users as the train keystrokes. The rest is validation keystrokes
    train_data = np.concatenate((correct_data[:n], other_data[:m]))
    train_labels = np.concatenate((correct_labels[:n], other_labels[:m]))
    valid_data = np.concatenate((correct_data[n:], other_data[m:]))
    valid_labels = np.concatenate((correct_labels[n:], other_labels[m:]))
    return train_data, train_labels, valid_data, valid_labels

#Shuffles two related arrays, preserving their correspondences.
def shuffle(data, labels):
    p = np.random.permutation(len(data))
    return data[p], labels[p]

util=Utilities()
def prepare_for_backprop(batch_size, Train_data, Train_labels, Valid_data, Valid_labels):
    print "Creating data..."
    batched_train_data, batched_train_labels = util.create_batches(Train_data, Train_labels,
                                              batch_size,
                                              create_bit_vector=True)
    batched_valid_data, batched_valid_labels = util.create_batches(Valid_data, Valid_labels,
                                              batch_size,
                                              create_bit_vector=True)
    print "Done!"


    return batched_train_data, batched_train_labels,  batched_valid_data, batched_valid_labels


if args.m==0: #Cross validation with knn
    test_data = np.concatenate((correct_data[args.train_correct:], other_data[args.train_other:]))
    test_labels = np.concatenate((correct_labels[args.train_correct:], other_labels[args.train_other:]))
    
    cross_data = np.concatenate((correct_data[:args.train_correct], other_data[:args.train_other]))
    cross_labels = np.concatenate((correct_labels[:args.train_correct], other_labels[:args.train_other]))
    cross_data, cross_labels = shuffle(cross_data, cross_labels)
    
    neighbors =  [1,2,3,4,5,6,7,8,9,10,15,20,50]
    FAR, FRR, FAR_avg, FRR_avg = cross_validation(cross_data, cross_labels, neighbors, 3)
 
    plt.plot(neighbors, FRR_avg, label="FRR")
    plt.plot(neighbors, FAR_avg, label="FAR")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel("False Acceptance/Rejectance rate")
    plt.xlabel("k")
    plt.title("k-Nearest Neighbors, on subject {}".format(args.subject))
    plt.show()
      
    nn=NearestNeighborClass()
    nn.train(cross_data, cross_labels)
    FA, FR = nn.predict(test_data, test_labels, 1)
    number_correct = np.count_nonzero(test_labels)
    number_incorrect = len(test_labels) - number_correct
    FAR = float(FA)/number_incorrect
    FRR = float(FR)/number_correct
    print FAR
    print FRR

elif args.m==1: #MLP anomaly detection
    correct_data, correct_labels = shuffle(correct_data, correct_labels)
    other_data, other_labels = shuffle(other_data, other_labels)
    Train_data = np.concatenate((correct_data[:args.train_correct], other_data[:args.train_other]))
    Train_labels = np.concatenate((correct_labels[:args.train_correct], other_labels[:args.train_other]))
    Valid_data = np.concatenate((correct_data[args.train_correct:], other_data[args.train_other:]))
    Valid_labels = np.concatenate((correct_labels[args.train_correct:], other_labels[args.train_other:]))

    batch_size=100;
    train_data, train_labels, valid_data, valid_labels=prepare_for_backprop(batch_size, Train_data, Train_labels, Valid_data, Valid_labels)

    mlp = MultiLayerPerceptron(layer_config=[31, 100, 100, 2], batch_size=batch_size)

    mlp.evaluate(train_data, train_labels, valid_data, valid_labels, eval_train=True)

    print("Done:)\n")

elif args.m==2: #Replicator neural net
    # The Number of correct/incorrect users in the training data
    number_correct = args.train_correct
    number_other = args.train_other
    # unbatched
    train_data, train_labels, valid_data, valid_labels = prepare_data(correct_data, correct_labels, other_data, other_labels, number_correct, number_other)
    batch_size=100;
    batch_train_data, batch_train_labels = util.create_batches(train_data, train_labels, batch_size, create_bit_vector=True)
    rnn = ReplicatorNeuralNet(layer_config=[31, 15, 15, 15, 31], batch_size=batch_size)
    loss = rnn.train(batch_train_data, epochs=70, learning_rate=0.0003)

    thresholds, far, frr = rnn.choose_outlier_thresholds(valid_data, valid_labels)

    plt.plot(loss)
    plt.xlabel("epoch")
    plt.ylabel("reconstruction loss")
    plt.title("Replicator neural network training on subject {}".format(args.subject))
    plt.show()

    # TODO make the plot nicer
    plt.plot(thresholds, far, label='FAR')
    plt.plot(thresholds, frr, label='FRR')
    plt.xlabel("threshold")
    plt.ylabel("rate")
    plt.title("Replicator neural network FAR/FRR on subject {}".format(args.subject))
    plt.show()

    # would need to use fresh data for this
    # actual_far, actual_frr = rnn.evaluate_error_rates(some_data, some_labels)
else:
    pass
