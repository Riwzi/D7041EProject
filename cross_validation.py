'''
Created on 6 mars 2017

@author: Rickard
'''
from nearestneighbor import *

def cross_validation(data, labels, neighbors, n=3):
    n=3
    #n-fold cross validation. Not really, since it's hard coded to 3 folds
    fold_size = len(data)/n
    folds_data = []
    folds_labels = []
    
    for i in range(n-1):
        folds_data.append(data[fold_size*i:fold_size*(i+1)])
        folds_labels.append(labels[fold_size*i:fold_size*(i+1)])
    #Add any leftovers to the final fold
    folds_data.append(data[fold_size*(n-1):])
    folds_labels.append(labels[fold_size*(n-1):])
    
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
        number_correct = np.count_nonzero(folds_labels[2])
        number_incorrect = len(folds_data[2]) - number_correct
        FAR1 = float(FA)/number_incorrect
        FRR1 = float(FR)/number_correct
        cross_FAR.append([k, FAR1])
        cross_FRR.append([k, FRR1])
 
        #Fold 0 is validation
        dat = np.concatenate((folds_data[1], folds_data[2]))
        lab = np.concatenate((folds_labels[1], folds_labels[2]))
        nn.train(dat, lab)
        FA, FR = nn.predict(folds_data[0], folds_labels[0], k)
        number_correct = np.count_nonzero(folds_labels[0])
        number_incorrect = len(folds_data[0]) - number_correct
        FAR2 = float(FA)/number_incorrect
        FRR2 = float(FR)/number_correct
        cross_FAR.append([k, FAR2])
        cross_FRR.append([k, FRR2])
 
        #Fold 1 is validation
        dat = np.concatenate((folds_data[0], folds_data[2]))
        lab = np.concatenate((folds_labels[0], folds_labels[2]))
        nn.train(dat, lab)
        FA, FR = nn.predict(folds_data[1], folds_labels[1], k)
        number_correct = np.count_nonzero(folds_labels[1])
        number_incorrect = len(folds_data[1]) - number_correct
        FAR3 = float(FA)/number_incorrect
        FRR3 = float(FR)/number_correct
        cross_FAR.append([k, FAR3])
        cross_FRR.append([k, FRR3])
 
        FAR = (FAR1+FAR2+FAR3)/3
        FRR = (FRR1+FRR2+FRR3)/3
        cross_FAR_avg.append(FAR)
        cross_FRR_avg.append(FRR)
    return cross_FAR, cross_FRR, cross_FAR_avg, cross_FRR_avg