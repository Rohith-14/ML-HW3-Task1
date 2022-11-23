# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 09:02:11 2022

@author: Lahari
"""

import numpy as np 
import pandas as pd 
import kmeans as km


dataset2=km.loadCSV('./Datasets/datam.csv')
k = len(set([x[0] for x in dataset2]))
print('k=',k)

# # Q1 - Comparing the SSE's of Euclidean-K-means, Cosine-K-means, Jarcard-K-means
# print('\n Q1 \n')
# clustering_euclidian = km.kmeans(dataset2,k,dist_type='euclidian')
# clustering_cosine = km.kmeans(dataset2,k,dist_type='cosine')
# clustering_jaccard = km.kmeans(dataset2,k,dist_type='jaccard')


# print('Euclidian: ', clustering_euclidian['withinss'])
# print('Cosine: ', clustering_cosine['withinss'])
# print('Jaccard: ', clustering_jaccard['withinss'])

# # Q2 - Comparing the accuracies of Euclidean-K-means, Cosine-K-means, Jarcard-K-means
# print('\n Q2 \n')
# def accuracy(cluster):
#     label_accuracy = list()
#     for cluster in cluster['clusters']:
#         labels = dict()
#         for item in cluster:
#             labels.setdefault(item[0],0)
#             labels[item[0]] += 1
#         vals = np.array(list(labels.values()))
#         vals.sort()
#         if len(vals) == 0:
#             label_accuracy.append(0)
#         else:
#             label_accuracy.append(vals[-1]/sum(vals))
#     label_accuracy = np.array(label_accuracy)
#     return label_accuracy.mean()

# print('Euclidian: ', accuracy(clustering_euclidian))
# print('Cosine: ', accuracy(clustering_cosine))
# print('Jaccard: ', accuracy(clustering_jaccard))

# # Q3 - Set up the same stop criteria: “when there is no change in centroid position"
# print(' \n Q3 \n')
# clustering_euclidian = km.kmeans(dataset2,k,dist_type='euclidian',condition='centroid')
# print('Euclidian: ', clustering_euclidian['iterations'])
# clustering_cosine = km.kmeans(dataset2,k,dist_type='cosine',condition='centroid')
# print('Cosine: ', clustering_cosine['iterations'])
# clustering_jaccard = km.kmeans(dataset2,k,dist_type='jaccard',condition='centroid')
# print('Jaccard: ', clustering_jaccard['iterations'])

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

