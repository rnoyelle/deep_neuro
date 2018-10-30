#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:04:37 2018

@author: jannes
"""
import numpy as np

def subset_train(train_labels, classes, size_of_batch):
    """Sorts and shuffles train_labels by class, then returns indices of size
    size_of_batch including equally-distributed indices from all classes.
    
    Args:
        train_labels: An ndarray. One-hot training labels.
        classes: An int. Number of unique classes.
        size_of_batch: An int. Batch size to train on. size_of_batch % classes
            must be 0.
            
    Returns:
        indices: A list. Indices to subset the data on before training.
    """
    batches = {}
    for i in range(classes):
        curr_ind = np.where(train_labels[:,i] == 1)
        np.random.shuffle(curr_ind[0])
        curr_batch = curr_ind[0][:int(size_of_batch/classes)]
        batches[i] = curr_batch
    indices = np.hstack(tuple(value for key, value in batches.items()))
    
    return indices

def subset_test(test_labels, classes):
    """Sorts and shuffles test_labels by class, then returns indices including 
    equally-distributed indices from all classes.
    
    Args:
        test_labels: An ndarray. One-hot test labels.
        classes: An int. Number of unique classes.
            
    Returns:
        indices: A list. Indices to subset the data on before testing.
    """
    values, counts = np.unique(test_labels, axis=0, return_counts=True)
    min_count = min(counts)
    
    batches = {}
    for i in range(classes):
        if test_labels.ndim > 1:
            curr_ind = np.where(test_labels[:,i] == 1)
        else:
            curr_ind = np.where(test_labels == i)
        np.random.shuffle(curr_ind[0])
        curr_batch = curr_ind[0][:min_count]  # use min_count samples per class
        batches[i] = curr_batch
    indices = np.hstack(tuple(value for key, value in batches.items()))
    
    return indices
