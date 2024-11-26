'''
This file contains the functions for training and evaluating a classifier on the learned representations.
'''

import numpy as np
import time as systime
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score


def fit_svm(features, y, MAX_SAMPLES=10000): # reuse from the softclt repo
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=10., gamma='scale')
    if train_size // nb_classes < 5 or train_size < 50:
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm, {
                'C': [
                    0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
                ],
                'kernel': ['rbf'],
                'gamma': ['scale', 'auto'],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [400],
                'class_weight': ['balanced'],
                'max_iter': [-1],
                'decision_function_shape': ['ovr'],
                'random_state': [131]
            },
            cv=3, n_jobs=-1
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            features, _, y, _ = train_test_split(features, y, train_size=MAX_SAMPLES, stratify=y, random_state=131)
        
        grid_search.fit(features, y)
        return grid_search.best_estimator_


def eval_classification(model, train_data, train_labels, test_data, test_labels):
    assert train_labels.ndim == 1 or train_labels.ndim == 2

    start_time = systime.time()
    train_repr = model.encode(train_data, **model.encode_args).detach().cpu().numpy()
    test_repr = model.encode(test_data, **model.encode_args).detach().cpu().numpy()
    inference_time = systime.time() - start_time
    num_samples = train_repr.shape[0] + test_repr.shape[0]

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_svm(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    return y_score, { 'acc': acc, 'auprc': auprc, 'inference_time': inference_time, 'num_samples': num_samples}

