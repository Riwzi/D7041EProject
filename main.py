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

util=Utilities()

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
#     parser.add_argument("train", help="Size of the training set", type=int) #these 3 doesn't do anything right now
#     parser.add_argument("valid", help="Size of the validation", type=int)
#     parser.add_argument("test", help="Size of the test set", type=int)
    parser.add_argument("subject", help="Id string of the correct subject.", type=str)

    # Optional Arguments
    parser.add_argument("-m", help="Method. 0 for KNN, 1 for RNN. Defaults to 0", type=int, default="0")

    # Parse arguments
    args = parser.parse_args()

    return args

args=parseArguments()
#data/labels for  the correct subjects and other subjects.
correct_data, correct_labels, other_data, other_labels = [], [], [], []
correct_subject = args.subject

#Read all data of the correct user into a separate array. Easier to split the data this way (i think)
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

correct_data = np.array(correct_data)
correct_labels = np.array(correct_labels)
other_data = np.array(other_data)
other_labels = np.array(other_labels)

#Convert all labels with correct subject to 1, all other labels to 0
correct_labels = np.where(correct_labels==correct_subject, np.ones_like(correct_labels), np.zeros_like(correct_labels))
other_labels = np.where(other_labels==correct_subject, np.ones_like(other_labels), np.zeros_like(other_labels)) #Gives an array of '' strings
other_labels[other_labels=='']='0' #Convert from '' to '0'.

def cross_validation(data1, labels1, data2, labels2, neighbors, n=3):
    n=3
    #n-fold cross validation. Not really, since it's hard coded to 3 folds
    fold_size1 = len(data1)/n
    fold_size2 = len(data2)/n
    folds_data = []
    folds_labels = []

    for i in range(n-1):
        data = np.concatenate((data1[fold_size1*i:fold_size1*(i+1)],data2[fold_size2*i:fold_size2*(i+1)]))
        folds_data.append(data)
        labels = np.concatenate((labels1[fold_size1*i:fold_size1*(i+1)], labels2[fold_size2*i:fold_size2*(i+1)]))
        folds_labels.append(labels)
    #Add any leftovers to the final fold
    data = np.concatenate((data1[fold_size1*(n-1):], data2[fold_size2*(n-1):]))
    folds_data.append(data)
    labels = np.concatenate((labels1[fold_size1*(n-1):], labels2[fold_size2*(n-1):]))
    folds_labels.append(labels)

    cross_FAR = []
    cross_FRR = []
    cross_k = []
    cross_FAR_avg = []
    cross_FRR_avg = []
    for k in neighbors:
        print ("K; ", k)
        cross_k.append(k)
        nn=NearestNeighborClass()

        #Fold 2 is validation
        dat = np.concatenate((folds_data[0], folds_data[1]))
        lab = np.concatenate((folds_labels[0], folds_labels[1]))
        nn.train(dat, lab)
        FA, FR = nn.predict(folds_data[2], folds_labels[2], k)
        FAR1 = float(FA)/fold_size2
        FRR1 = float(FR)/fold_size1
        cross_FAR.append([k, FAR1])
        cross_FRR.append([k, FRR1])

        #Fold 0 is validation
        dat = np.concatenate((folds_data[1], folds_data[2]))
        lab = np.concatenate((folds_labels[1], folds_labels[2]))
        nn.train(dat, lab)
        FA, FR = nn.predict(folds_data[0], folds_labels[0], k)
        FAR2 = float(FA)/fold_size2
        FRR2 = float(FR)/fold_size1
        cross_FAR.append([k, FAR2])
        cross_FRR.append([k, FRR2])

        #Fold 1 is validation
        dat = np.concatenate((folds_data[0], folds_data[2]))
        lab = np.concatenate((folds_labels[0], folds_labels[2]))
        nn.train(dat, lab)
        FA, FR = nn.predict(folds_data[1], folds_labels[1], k)
        FAR3 = float(FA)/fold_size2 #might be slightly wrong here
        FRR3 = float(FR)/fold_size1
        cross_FAR.append([k, FAR3])
        cross_FRR.append([k, FRR3])

        FAR = (FAR1+FAR2+FAR3)/3
        FRR = (FRR1+FRR2+FRR3)/3
        cross_FAR_avg.append(FAR)
        cross_FRR_avg.append(FRR)
    return cross_FAR, cross_FRR, cross_FAR_avg, cross_FRR_avg


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
    cross_correct_data = correct_data[:300]
    cross_correct_labels = correct_labels[:300]
    cross_other_data = other_data[:15000]
    cross_other_labels = other_labels[:15000]
    neighbors =  [1,2,3,4,5,6,7,8,9,10,15,20,50]
    FAR, FRR, FAR_avg, FRR_avg = cross_validation(cross_correct_data, cross_correct_labels, cross_other_data, cross_other_labels, neighbors, 3)
    print FAR
    print FAR_avg
    print FRR
    print FRR_avg

    plt.plot(neighbors, FRR_avg, label="FRR")
    plt.plot(neighbors, FAR_avg, label="FAR")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel("False Acceptance/Rejectance rate")
    plt.xlabel("k")
    plt.title("k-Nearest Neighbors")
    plt.show()

elif args.m==1: #MLP anomaly detection
    Train_data = np.concatenate((correct_data[:300], other_data[:15000]))
    Train_labels =np.concatenate((correct_labels[:300], other_labels[:15000]))
    Valid_data = np.concatenate((correct_data[300:], other_data[15000:]))
    Valid_labels = np.concatenate((correct_labels[300:], other_labels[15000:]))

    #Convert labels to ints instead of strings because utilities kept complaining. Does it matter? idk
    Train_labels = np.array(map(int, Train_labels))
    Valid_labels = np.array(map(int, Train_labels))

    batch_size=100;
    train_data, train_labels, valid_data, valid_labels=prepare_for_backprop(batch_size, Train_data, Train_labels, Valid_data, Valid_labels)

    mlp = MultiLayerPerceptron(layer_config=[31, 100, 100, 2], batch_size=batch_size)

    mlp.evaluate(train_data, train_labels, valid_data, valid_labels, eval_train=True)

    print("Done:)\n")

else:
    pass
