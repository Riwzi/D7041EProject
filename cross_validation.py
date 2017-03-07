'''
Created on 6 mars 2017

@author: Rickard
'''
from NearestNeighbor import *

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