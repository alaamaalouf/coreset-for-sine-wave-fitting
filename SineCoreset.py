import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import time
import multiprocessing
from multiprocessing import Process
import psutil
import scipy.io as sio
# from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde

REPS = 20
EULER_SQUARED_SINE = lambda x: np.exp(x)
CORES_IN_MACHINE = psutil.cpu_count(logical=False)
# EULER_SQUARED_SINE = lambda x: -1.0/4.0 * (np.exp(x) - np.exp(-x)) ** 2

def computeSense(P, N=int(5e3)):
    # loss = lambda p,x: np.sin(2*np.pi/N * p * x) ** 2
    Q, _, counts = np.unique(P, return_counts=True, return_index=True)
    s_func = lambda p, x,c: c*np.sin(2*np.pi*p*x/N) ** 2 / np.sum(np.multiply(np.sin(2*np.pi * Q*x/N)**2,counts))

    max_x = np.empty((Q.shape[0],))
    sens = dict()
    s = np.empty((P.shape[0], ))
    for i in range(Q.shape[0]):
        max_s = -np.inf
        for x in range(1, int(N)):
            temp = s_func(Q[i], x, counts[i])
            if temp > max_s:
                max_s = temp
                sens[Q[i]] = max_s
                max_x[i] = x

    for i in range(len(P)):
        s[i] = sens[P[i]]
    return s, sens, max_x

def sensitivty_sequental(Q,s_func, sens, start, end,N):
    #print(start,end)
    max_x = np.empty((Q.shape[0],))

    for i in range(start, end,1):
        t = time.time()
        max_s = -np.inf
        for x in range(1, int(N)):
            temp = s_func(Q[i], x)
            if temp > max_s:
                max_s = temp
                sens[Q[i]] = max_s
                max_x[i] = x
        #print(time.time() -t)

    return sens
def computeSenseParallel(P, N, procs):
    manager = multiprocessing.Manager()
    sens = manager.dict()
    jobs = []

    Q, idxs, counts = np.unique(P, return_counts=True, return_index=True)
    s_func = lambda p, x: np.sin(2 * np.pi * p * x / N) ** 2 / np.sum(np.multiply(np.sin(2 * np.pi * Q * x / N) ** 2, counts))
    starters = range(0, Q.shape[0], int(np.ceil(Q.shape[0]/procs)))

    for idx in range(procs):
        porcces = Process(target=sensitivty_sequental,
                          args=(Q, s_func, sens,
                                starters[idx] ,
                                min(starters[idx] + int(np.ceil(Q.shape[0]/procs)) , Q.shape[0]),
                                N
                                )
                          )

        jobs.append(porcces)
        porcces.start()
    for porcces in jobs: porcces.join()

    s = np.empty((P.shape[0],))
    #print(len(sens.keys()))
    for i in range(len(P)):
        s[i] = sens[P[i]]
    return s

def sensitivty_sequental_via_look_up_table(Q,look_up, sens, start, end,N,counts):
    max_x = np.empty((Q.shape[0],))
    #print (end)
    for i in range(start, end, 1):
        #t = time.time()
        if i%500 ==0 : print (start,i,end)
        max_s = -np.inf
        for x in range(1, int(N)):
            temp = counts[i] * np.sin(2 * np.pi * Q[i] * x / N) ** 2 /look_up[x-1]
            if temp > max_s:
                max_s = temp
                sens[Q[i]] = max_s
                max_x[i] = x
        #print(time.time() - t)

    return sens

def computeSenseParallelViaLookUpTable(P,N,procs,w=None):
    manager = multiprocessing.Manager()
    sens = manager.dict()
    jobs = []

    Q, idxs, counts = np.unique(P, return_counts=True, return_index=True)
    if w is not None:
        s_func = lambda x: np.sum(np.multiply(np.sin(2 * np.pi * Q * x / N) ** 2, w))
    else:
        s_func = lambda x: np.sum(np.multiply(np.sin(2 * np.pi * Q * x / N) ** 2, counts))

    look_up = [s_func(x) for x in range(1, int(N))]
    starters = range(0, Q.shape[0], int(np.ceil(Q.shape[0]/procs)))

    for idx in range(procs):
        porcces = Process(target=sensitivty_sequental_via_look_up_table,
                          args=(Q, look_up, sens,
                                starters[idx] ,
                                min(starters[idx] + int(np.ceil(Q.shape[0]/procs)) , Q.shape[0]),
                                N, counts))
        jobs.append(porcces)
        porcces.start()
    for porcces in jobs: porcces.join()

    s = np.empty((P.shape[0],))
    for i in range(len(P)): s[i] = sens[P[i]]
    return s

def computeCoreset(P, s,w, sample_size):
    t = np.sum(s)
    prob = s/t
    # np.random.seed(48)
    indices = np.random.choice(np.arange(P.shape[0]), sample_size, p=prob)
    unique, counts = np.unique(indices, return_counts=True)
    return P[unique].astype(np.int), \
           np.multiply(np.multiply(np.ones((unique.shape[0])), counts), w[unique] / prob[unique]) / sample_size











