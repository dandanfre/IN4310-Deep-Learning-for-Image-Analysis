import os, sys, numpy as np

import torch

import time

from torch.autograd import Variable


def forloopdists(feats, protos):
    N, D = feats.shape
    P, _ = protos.shape
    distance = np.zeros((N, P))  # initialize the distance matrix
    for i in range(N):
        for j in range(P):
            distance[i, j] = np.sum(
                (feats[i] - protos[j]) ** 2
            )  # squared Euclidean distance


def numpydists(feats, protos):
    N, D = feats.shape
    P, _ = protos.shape
    distance = np.zeros((N, P))  # initialize the distance matrix
    for i in range(N):  # loop over rows of X
        diff = feats[i, np.newaxis, :] - protos  # Broadcast X[i] over T
        distance[i, :] = np.sum(
            (feats[i] - protos) ** 2, axis=1
        )  # squared Euclidean distance


# def pytorchdists(feats0, protos0, device=None):


# YOUR implementation here


def run():

    ########
    ##
    ## if you have less than 8 gbyte, then reduce from 250k
    ##
    ###############
    feats = np.random.normal(
        size=(250000, 300)
    )  # 5000 instead of 250k for forloopdists
    protos = np.random.normal(size=(500, 300))

    """
  since = time.time()
  dists0=forloopdists(feats,protos)
  time_elapsed=float(time.time()) - float(since)
  print('Comp complete in {:.3f}s'.format( time_elapsed ))
  """

    device = torch.device("cpu")
    since = time.time()

    dists1 = numpydists  # pytorchdists(feats, protos, device)

    time_elapsed = float(time.time()) - float(since)

    print("Comp complete in {:.3f}s".format(time_elapsed))
    print(dists1.shape)

    # print('df0',np.max(np.abs(dists1-dists0)))

    since = time.time()

    dists2 = numpydists(feats, protos)

    time_elapsed = float(time.time()) - float(since)

    print("Comp complete in {:.3f}s".format(time_elapsed))

    print(dists2.shape)

    print("df", np.max(np.abs(dists1 - dists2)))


if __name__ == "__main__":
    run()
