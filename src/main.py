import os
import sys
import numpy as np
import argparse
from utils import detect_lof, detect_isoforest
from outlier_interpreter import OutlierInterpreter


def main(args):
    # read data
    num_inst, dim, X = read_data(args)
    X = np.array(X, dtype=float)

    # measure degree of outlierness
    labels_otlr = detect_outliers(args, X)

    # interpret outliers
    sgnf_prior = 1
    interpreter = OutlierInterpreter(X, labels_otlr, args.ratio_nbr,
                                     AUG=args.AUG, MIN_CLUSTER_SIZE=args.MIN_CLUSTER_SIZE, MAX_NUM_CLUSTER=args.MAX_NUM_CLUSTER,
                                     VAL_TIMES = args.VAL_TIMES, C_SVM=args.C_SVM, THRE_PS=args.THRE_PS, DEFK=args.DEFK)
    ids_target = np.where(labels_otlr == 1)[0]      # sample id of outliers
    importance_attr, outlierness = interpreter.interpret_outliers(ids_target, sgnf_prior, int_flag=1)

    print("Sample ID of outliers:", '\n', ids_target)
    print("Outlying degree of attributes:", '\n', importance_attr)
    print("Outlierness scores re-estimated:", '\n', outlierness)

    return ids_target, importance_attr, outlierness


def read_data(args):
    data_name = args.dataset
    fn = os.path.join(os.path.dirname(os.getcwd()), "data", data_name, "X.csv")
    X = np.genfromtxt(fn, delimiter=',', dtype=int)
    num_inst = X.shape[0]
    dim = X.shape[1]

    return num_inst, dim, X


def detect_outliers(args, X):
    # A larger outlier score means greater outlierness
    if args.detector == 'lof':
        labels_otlr = detect_lof(args, X)
        labels_otlr = np.array(0.5*(1-labels_otlr), dtype=int)      # outlier label = 1, normal label = 0
        #print(labels_otlr)
    elif args.detector == 'isoforest':
        labels_otlr = detect_isoforest(args, X)
        labels_otlr = np.array(0.5 * (1 - labels_otlr), dtype=int)  # outlier label = 1, normal label = 0
        #print(labels_otlr)
    else:
        print("The detector type is not considered in current implementation...")
        sys.exit()

    return labels_otlr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wbc', help='which dataset to use')
    parser.add_argument('--ratio_nbr', type=float, default=0.1,
                        help='controls number of neighbors to use in kneighbors queries')
    parser.add_argument('--detector', type=str, default='lof', help='which outlier detector to use')
    parser.add_argument('--AUG', type=float, default=10, help='an additional attribute value as augmentation')
    parser.add_argument('--MIN_CLUSTER_SIZE', type=int, default=5,
                        help='minimum number of samples required in a cluster')
    parser.add_argument('--MAX_NUM_CLUSTER', type=int, default=4, help='maximum number of clusters for each context')
    parser.add_argument('--VAL_TIMES', type=int, default=10, help='number of iterations for computing prediction strength')
    parser.add_argument('--C_SVM', type=float, default=1., help='penalty parameter for svm')
    parser.add_argument('--DEFK', type=int, default=0,
                        help='pre-determined number of clusters in each context (use prediction strength if 0)')
    parser.add_argument('--THRE_PS', type=float, default=0.85,
                        help='threshold for deciding the best cluster value in prediction strength')
    parser.add_argument('--RESOLUTION', type=float, default=0.05, help='attribute resolution')
    args = parser.parse_args()

    main(args)