'''
Created on Dec 1, 2016

@author:  
Adopted from CS231n
'''

import numpy as np
import progressbar

class NearestNeighborClass(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N. """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, Y ,k=1):
        """ X is N x D where each row is an example we wish to predict label for """
        """ Y is 1-dimension of size N where each row is the label for the corresponding example in X """
        num_test = X.shape[0]
        bar = progressbar.ProgressBar(maxval=num_test, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # False Acceptance/False Rejection counters
        FA = 0
        FR = 0
        
        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training keystroke to the i'th test keystroke
            # using the L1 distance (sum of absolute value differences)
            # distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            # using the L2 distance (sum of square value differences)
            distances = np.sum(np.square(self.Xtr - X[i,:]), axis = 1)
            
#             min_index = np.argmin(distances) # get the index with smallest distance
#             Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
            min_indices = np.argpartition(distances, k)[:k]
            min_labels = self.ytr[min_indices]
            labels, count = np.unique(min_labels, return_counts=True)
            Ypred[i] = labels[count.argmax()]
            
            prediction = labels[count.argmax()]
            if (prediction == '1'):
                if (Y[i] == '1'):
                    # It was the correct user and he/she was accepted
                    pass
                else:
                    # It was an incorrect user and he/she was accepted
                    FA += 1
            else:
                if (Y[i] == '1'):
                    # it was the correct user and he/she was rejected
                    FR += 1
                else:
                    # It was an incorrect user and he/she was rejected
                    pass
            
            bar.update(i+1)
        bar.finish()

        return FA, FR

