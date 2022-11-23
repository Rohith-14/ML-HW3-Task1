# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 09:02:11 2022

@author: Rohith
"""

import numpy as np 
import pandas as pd 
import kmeans as km


dataset2=km.loadCSV('./Datasets/datam.csv')
k = len(set([x[0] for x in dataset2]))
print('k=',k)

print(' \n Q4 \n')
from time import time

def run_condition(condition, dataset,k):

    euclidian_start = time()
    clustering_euclidian = km.kmeans(dataset,k,dist_type='euclidian',condition=condition)
    euclidian_time = time() - euclidian_start
    print('Euclidian SSE: ', clustering_euclidian['withinss'])

    print("Euclidian \t Time: {} \t Iterations: {}".format(euclidian_time, clustering_euclidian['iterations']))
    
    cosine_start = time()
    clustering_cosine = km.kmeans(dataset,k,dist_type='cosine',condition=condition)
    cosine_time = time() - cosine_start
    print('Cosine SSE: ', clustering_cosine['withinss'])

    print("Cosine \t\t Time: {} \t Iterations: {}".format(cosine_time, clustering_cosine['iterations']))

    jaccard_start = time()
    clustering_jaccard = km.kmeans(dataset,k,dist_type='jaccard',condition=condition)
    jaccard_time = time() - jaccard_start
    print('Jaccard SSE: ', clustering_jaccard['withinss'])
    print("Jaccard \t Time: {} \t Iterations: {}".format(euclidian_time, clustering_jaccard['iterations']))
    return '\n'

print('Termination condition: when there is no change in centroid position')
print(run_condition('centroid',dataset2,k))
print('\n')
print('Termination condition: when the SSE value increases in the next iteration')
print(run_condition('sse',dataset2,k))
print('Termination condition: when the maximum preset value (100) of iteration is complete')
print(run_condition('iteration',dataset2,k))

