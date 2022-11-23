# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:50:38 2022

@author: Rohith Neralla
"""

import numpy as np
import pandas as pd
#from sklearn.cluster import KMeans
import scipy 
import sklearn
from collections import Counter
from sklearn.metrics import multilabel_confusion_matrix
from scipy import spatial


data = pd.read_csv('data.csv')
labels = pd.read_csv('label.csv',names=['label'],header=None)
print(data.head())

#Splitting the data data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split( data, test_size=0.08, random_state=42)
train_labels, test_labels = train_test_split( labels, test_size=0.08, random_state=42)

class KMeans:
    
    def calculate_SSE(self, centroid_value_dict, centroidDict,data):
        sse_data = 0
        for i in centroidDict:
            sse_clust = 0
            # np.sum()
            for j in centroidDict[i]:
                dp = list(data.iloc[int(j)])
                for a,b in zip(centroid_value_dict[i],dp):
                    sse_clust += (a-b)**2
            sse_data+=sse_clust
        return sse_data    
    
    def Initialize_Centroids(self,data,K):
        m = data.shape[0]
        centroid_value_dict={}
        for i in range(K):
            r = np.random.randint(0, m-1)
            centroid_value_dict[i] = data.iloc[r]
        return centroid_value_dict
    
    def jaccard_similarity(self,centroid, dp):
        intersection = len(list(set(centroid).intersection(dp)))
        union = (len(set(centroid)) + len(set(dp))) - intersection
        return float(intersection) / union

    def train_Kmeans(self,data,K,max_iter=20,mode=1,tol=10):
        #Mode = 1 => Euclidean np.linalg.norm(x-list(data.iloc[i,:]))
        #Mode = 2 => Jaccard
        #Mode = 3 => Cosine
        centroid_value_dict = self.Initialize_Centroids(data,K)
        new_centroid_value_dict = {}
        count = 0
        centroidDict = {}
        convergence = False
        while((count<max_iter) and not convergence):
            
            for i in list(centroid_value_dict.keys()):
                centroidDict[i]=[]
            for i in range(data.shape[0]):
                x = data.iloc[i]
                if mode==1 :
                    distance_measure = [np.linalg.norm(x-centroid_value_dict[j])  for j in centroid_value_dict]
                    idx = np.argmin(distance_measure)
                    centroidDict[idx].append(i)
                elif mode==2 :
                    distance_measure = [self.jaccard_similarity(list(x),centroid_value_dict[j]) for j in centroid_value_dict]
                    idx = np.argmax(distance_measure)
                    centroidDict[idx].append(i)
                elif mode==3 :
                    distance_measure = [1-scipy.spatial.distance.cosine(x,list(centroid_value_dict[j]))  for j in centroid_value_dict]
                    idx = np.argmax(distance_measure)
                    centroidDict[idx].append(i)
                
                prev_centroids=dict(centroid_value_dict)
                
            
            for i in centroidDict:
                if len(centroidDict[i]):
                    dps_centroid = centroidDict[i]
                    centroid_value_dict[i] = np.average(data.iloc[dps_centroid],axis=0)
            
            
            current_tol=-1
            for i in centroid_value_dict:
                prev_centroid_point = prev_centroids[i]
                new_centroid_point = centroid_value_dict[i]
                change = np.sum(np.absolute(new_centroid_point-prev_centroid_point))
                current_tol = max(change, current_tol)
                
            print("Tolerance for the Iteration ",count,": ",current_tol)
            
            count+=1
            if (current_tol<10):
                convergence = True
                break
           # print("KMeans Iteration",count)
        return centroid_value_dict,centroidDict
    
def predict_cluster_labels(C, S, labels):
    '''
    Input : C -> Centroids
            S -> Set of Indicies corresponding to Centroid C
            data -> Data used to form clusters
    Output : Returns an array of size K having labels based on majority voting in the cluster
    '''
    cluster_labels = np.zeros(10,dtype=int)
    for c in C:
        labels_of_points = []
        for point in S[c]:
            labels_of_points.extend(labels.iloc[point])
        counter = Counter(labels_of_points)
        try:
            cluster_labels[c] = max(counter, key=counter.get)
        except:
            cluster_labels[c] = np.random.randint(0,9)
    return cluster_labels

def jaccard_similarity(centroid, dp):
        intersection = len(list(set(centroid).intersection(dp)))
        union = (len(set(centroid)) + len(set(dp))) - intersection
        return float(intersection) / union
    
def accuracy(centroids, centroid_Labels, test_data, true_labels, mode=1):
    y_true = list(true_labels['label']);
    y_pred = []
    for index in range(test_data.shape[0]):
        featureset = test_data.iloc[index]
        if mode==1:
            distances = [np.linalg.norm(featureset - centroids[centroid]) for centroid in centroids]
            classification = distances.index(min(distances))
            y_pred.append(centroid_Labels[classification])
        elif mode==2:
            similarity = [jaccard_similarity(featureset, centroids[centroid]) for centroid in centroids]
            classification = similarity.index(max(similarity))
            y_pred.append(centroid_Labels[classification]) 
        elif mode==3:
            similarity = [1 - spatial.distance.cosine(featureset, centroids[centroid]) for centroid in centroids]
            classification = similarity.index(max(similarity))
            y_pred.append(centroid_Labels[classification])
    denominator = test_data.shape[0]
    correctly_classified = 0
    for i in range(0,len(y_pred)):
        if y_true[i] == y_pred[i]:
            correctly_classified += 1
    accuracy = correctly_classified/denominator
    return accuracy

model1 = KMeans()
print('Eucledian iterations')
centroids1,clusters1 = model1.train_Kmeans(data,10, max_iter=100,mode=1)
Euclidean_SSE = model1.calculate_SSE(centroids1,clusters1,data)
print("Euclidean SSE:",Euclidean_SSE)
cluster_labels1 = predict_cluster_labels(centroids1,clusters1,labels)
cluster_labels1
Accuracy_Euclidean = accuracy(centroids1, cluster_labels1,test_data,test_labels)
print("Euclidean Accuracy:",Accuracy_Euclidean)

model2 = KMeans()
print('Jaccard iterations')
centroids2,clusters2 = model2.train_Kmeans(data,10, max_iter=100,mode=2)
Jaccard_SSE = model2.calculate_SSE(centroids2,clusters2,data)
print("Jaccard SSE:",Jaccard_SSE)
cluster_labels2 = predict_cluster_labels(centroids2,clusters2,labels)
cluster_labels2
Accuracy_Jaccard = accuracy(centroids2, cluster_labels2,test_data,test_labels)
print("Jaccard accuracy:",Accuracy_Jaccard)

model3 = KMeans()
print('Cosine iterations')
centroids3,clusters3 = model3.train_Kmeans(data,10, max_iter = 100,mode=3)
Cosine_SSE = model3.calculate_SSE(centroids3,clusters3,data)
print("Cosine SSE :",Cosine_SSE)
cluster_labels3 = predict_cluster_labels(centroids3,clusters3,labels)
cluster_labels3
Accuracy_Cosine = accuracy(centroids3, cluster_labels3,test_data,test_labels)
print("Cosine accuracy :",Accuracy_Cosine)


print('\n SSEs \n')
print("Euclidean SSE:",Euclidean_SSE)
print("Jacard SSE:",Jaccard_SSE)
print("Cosine SSE :",Cosine_SSE)

print('\n Accuracies \n')
print("Euclidean accuracy:",Accuracy_Euclidean)
print("Jacard accuracy:",Accuracy_Jaccard)
print("Cosine accuracy :",Accuracy_Cosine)

