'''
Created on 3 Jan 2017

@author: eao
'''
import numpy as np
import os
import sys

def label_to_bit_vector(labels, nbits):
    bit_vector = np.zeros((labels.shape[0], nbits))
    for i in range(labels.shape[0]):
        bit_vector[i, labels[i]] = 1.0

    return bit_vector

class Utilities(object):
    '''
    classdocs
    '''
    def test(self):

        #       dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.getcwd()
        print("Util.Test is done. DIR=", dir_path)
        return
    
    def exit_with_error(self, err_str):
        print >> sys.stderr, err_str
        sys.exit(1)
    
    def create_batches(self, data, labels, batch_size, create_bit_vector=False):
        N = data.shape[0]
        if N % batch_size != 0:
            print "Warning in create_minibatches(): Batch size {0} does not " \
              "evenly divide the number of examples {1}.".format(batch_size,N)
        chunked_data = []
        chunked_labels = []
        idx = 0
        while idx + batch_size <= N:
            chunked_data.append(data[idx:idx+batch_size, :])
            if not create_bit_vector:
                chunked_labels.append(labels[idx:idx+batch_size])
            else:
                bit_vector = label_to_bit_vector(labels[idx:idx+batch_size], 2)
                chunked_labels.append(bit_vector)

            idx += batch_size

        return chunked_data, chunked_labels


    
    def __init__(self):
        '''
        Constructor
        '''
        