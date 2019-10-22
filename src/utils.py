import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def detect_lof(args, X):
    num_inst = X.shape[0]
    num_nbr = int(num_inst * args.ratio_nbr)
    clf = LocalOutlierFactor(n_neighbors=num_nbr)
    y_pred = clf.fit_predict(X)
    outlier_scores = -clf.negative_outlier_factor_

    return y_pred


def detect_isoforest(args, X):
    num_inst = X.shape[0]
    clf = IsolationForest(behaviour='new', max_samples=num_inst, random_state=0)
    clf.fit(X)
    y_pred = clf.predict(X)
    outlier_scores = -clf.decision_function(X)

    return y_pred