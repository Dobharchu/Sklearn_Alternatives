"""
In this file are some useful functions similar to those in scikitlearn
"""


def split(dataset, labels, seed = None, test_size = 0.2, stratify = True):
    '''
    A function that takes in a dataset and splits it into training set, 
    training labels, test set, test labels.
    Example: 
    train_x, train_y, test_x, test_y = split(x, y, seed = 45,
                                             test_size = 0.2, stratify = True)
    dataset -> is a pandas dataset
    labels -> are the name(s) of the columns that contain the labels
    seed -> is the seed to set for random
    test_size -> is the proportion of the data dedicated to the test set
    stratify -> stratify the data

    '''

    import random

    if stratify == False: # not done yet
        print('Not done yet')
        return
    
    if seed is not None:
        random.seed(seed)

    if stratify == True:
        """
        How this works is by basically creating a list of indexes for 
        each label in the dataset. (Separating them by label)
        Then based on the test ratio (size) we sample each of these 
        labels individually beforecombining them back into test and 
        train sets, hopefully preserving ratios between test and train
        sets.
        """
        unique = np.unique(labels) #creating an array of unique elements
        #creating a list of indexes
        x = [np.where(labels == i)[0] for i in unique]
        index_list_test = []
        for i in x:
            n = len(i)
            n_test = int(test_size *n)
            #Combining all the indexes designated for test set into one list
            index_list_test = index_list_test + random.sample(list(i),
                                                               k = n_test)

        test_set = dataset[[index_list_test]]
        test_labels = labels[[index_list_test]]

        #Getting the indexes for the train set
        index_list_train = [i for i in list(range(0,len(dataset)))
                             if i not in index_list_test]
        train_set = dataset[[index_list_train]]
        train_labels = labels[[index_list_train]]

    return train_set[0], train_labels[0], test_set[0], test_labels[0]


def calc_weights_balanced(labels):
    '''
    This function returns a dictionary of balanced weights from a labels
    dataframe that can be used during the model.fit of tensorflow.
    '''

    import numpy as np
    u, counts = np.unique(labels, return_counts = True)
    n = len(labels)
    weights = {}
    for i in range(len(u)):
        #This is the calculation used for balanced according to scikitlearn
        weight = (n / (len(u) * counts[i])) 
        weights[u[i]] =  weight

    return weights
