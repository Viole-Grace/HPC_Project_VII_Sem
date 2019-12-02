#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numba
import numpy as np
import argparse
import time


# In[145]:


run_parallel = numba.config.NUMBA_NUM_THREADS > 1

@numba.njit(parallel=run_parallel)
def linear_regression(Y, X, w, iterations, alphaN):
    for i in range(iterations):
        w -= alphaN * np.dot(X.T, np.dot(X,w)-Y)
    return w

def parallel_main():
    
    N = 20000
    D = 10
    p = 4
    iterations = 20
    alphaN = 0.01/N
    w = np.zeros((D,p))
    np.random.seed(0)
    points = np.random.random((N,D))
    labels = np.random.random((N,p))
    t1 = time.time()
    w = linear_regression(labels, points, w, iterations, alphaN)
    selftimed = time.time()-t1
    print("Bias values (parallel) : \n\n{}\n".format(w))
    print("Parallel Execution time (seconds) ", selftimed)
    print("checksum: ", np.sum(w),"\n\n")
    return np.sum(w), selftimed


# In[150]:


def linear_regression(Y, X, w, iterations, alphaN):
    for i in range(iterations):
        w -= alphaN * np.dot(X.T, np.dot(X,w)-Y)
    return w

def serial_main():
    
    N = 20000
    D = 10
    p = 4
    iterations = 20
    alphaN = 0.01/N
    w = np.zeros((D,p))
    np.random.seed(0)
    points = np.random.random((N,D))
    labels = np.random.random((N,p))
    
    print("Dataset size : {}".format(N))
    print("Dataset : \n",points[:10])
    
    t1 = time.time()
    w = linear_regression(labels, points, w, iterations, alphaN)
    selftimed = time.time()-t1
    print("\nBias values (serial) : \n\n{}\n".format(w))
    print("Serial Execution time (seconds) ", selftimed)
    print("checksum: ", np.sum(w))
    return np.sum(w), selftimed


# In[196]:


checksum_s, stime = serial_main()
checksum_p, ptime = parallel_main()

print("Speedup :{:.2f}%".format(((stime-ptime)/stime)*100))
if checksum_p == checksum_s:
    print("Successful Execution")


# In[ ]:




