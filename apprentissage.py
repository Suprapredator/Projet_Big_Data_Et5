# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
from sklearn.neighbors import KNeighborsClassifier;
import time;

def getSklearnInstance(data):
    X = list()
    Y = list()    
    
    instance = KNeighborsClassifier(n_neighbors=10, weights='uniform')
    nx, ny, npixel, nombre_d_elements = data['X'][:,:,:,:].shape
    #tampon = data['X'][:,:,:,:].reshape(nombre_d_elements,nx*ny*npixel)    
    
    for i in range(15000):
        Y.append(data['y'][i])
        X.append(data['X'][:,:,:,i].reshape(nx*ny*npixel))
    
    instance.fit(X, np.ravel(Y))
    
    return instance
    
def donneMoiLaClasse(X, SklearnInstance):
    nx, ny, npixel = X.shape
    return SklearnInstance.predict(X.reshape(1,nx*ny*npixel)) 
    
def test(train_data, test_data, taille):
    start_time = time.time()
    bonneReponse = 0;
    SklearnInstance = getSklearnInstance(train_data)

    start_time2 = time.time()
    
    if taille == -1 or taille > len(test_data['y']):
        taille = len(test_data['y'])
    
    for i in range(taille):
        if test_data['y'][i]%10 == donneMoiLaClasse(test_data["X"][:, :, :, i], SklearnInstance):
            bonneReponse += 1
    
    print('Résultat par Sklearn: '+str(bonneReponse)+'/'+str(taille))
    print("--- temps creation instance = %s seconds ---" % (start_time2 - start_time))
    print("--- temps recherche voisin = %s seconds ---" % (time.time() - start_time2))
    print("--- temps total = %s seconds ---" % (time.time() - start_time))
