import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


def expand_data(data, n_ref=2, dist_perturb=0.5):
    n_sample = data.shape[0]
    n_feat = data.shape[1]
    data_expand = np.zeros([n_sample, n_feat])

    nbrs = NearestNeighbors(n_neighbors=n_ref).fit(data)
    for n in range(n_sample):
        ind = np.random.randint(0, n_sample)
        point_org = [data[ind]]
        distNbs, _ = nbrs.kneighbors(point_org)

        perturb = np.mean(distNbs[0]) * dist_perturb
        shift_direction = np.random.random(data.shape[1])
        shift_direction /= np.linalg.norm(shift_direction)  # normalize

        data_expand[n] = data[ind] + perturb*shift_direction

    return data_expand


def ClosestCenter(point, centroids):
    # Find the closest center over all centroids
    min_index = -1
    min_dist = float('inf')
    for i in range(len(centroids)):
        center = centroids[i]
        dist_cur = np.linalg.norm(point - center)
        if dist_cur < min_dist:
            min_index = i
            min_dist = dist_cur

    return min_index


def PredictionStrength(data_test, test_labels, train_centers, c):
    # Compute prediction strength under c clusters
    pred_strength = np.zeros(c)
    for cc in range(c):
        num_cc = test_labels.tolist().count(cc)
        count = 0.
        for i in range(len(test_labels)-1):
            for j in range(i+1, len(test_labels)):
                if test_labels[i] == test_labels[j] == cc:
                    pi = data_test[i]
                    pj = data_test[j]
                    if ClosestCenter(pi, train_centers) == ClosestCenter(pj, train_centers):
                        count += 1

        if num_cc <= 1:
            pred_strength[cc] = float('inf')
        else:
            pred_strength[cc] = count/(num_cc * (num_cc-1)/2.)

    return min(pred_strength)


def optimalK(data, num_fold, maxClusters=5, THRE_PS=0.90):
    # Find the best number of clusters using prediction strength
    num_data = data.shape[0]
    num_feat = data.shape[1]

    pred_strength_avg = np.zeros(maxClusters+1)
    for nf in range(num_fold):
        # Split into training and testing samples
        if (num_data/num_feat < 5):   # data is sparse, use synthetic sampling
            print("Context may be too sparse...")
        inds_train = np.random.choice(num_data, int(num_data*0.8), replace=False)
        inds_test = list(set(range(num_data)).difference(inds_train))
        data_train = data[inds_train]
        data_test = data[inds_test]

        pred_strength_cur = np.zeros(maxClusters+1)
        for c in range(1, maxClusters+1):
            train_cluster = KMeans(n_clusters=c).fit(data_train)
            test_cluster = KMeans(n_clusters=c).fit(data_test)
            pred_strength_cur[c] = PredictionStrength(data_test, test_cluster.labels_, train_cluster.cluster_centers_, c)

        pred_strength_avg += pred_strength_cur

    pred_strength_avg /= num_fold
    # print("Prediction Strength vec: ", pred_strength_avg)

    k_optimal = max([i for i,j in enumerate(pred_strength_avg) if j > THRE_PS])

    return k_optimal


if __name__ == "__main__":
    x, y = make_blobs(24, n_features=5, centers=3)
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

    k = optimalK(x, 10)
    print('Optimal k is: ', k)