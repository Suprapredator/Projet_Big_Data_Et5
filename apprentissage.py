# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;
import math as mt;
from sklearn.neighbors import NearestNeighbors;

def getSklearnInstance(data):
    instance = NearestNeighbors()
    nx, ny, npixel, nombre_d_elements = data['X'][:,:,:,:].shape
    tampon = data['X'][:,:,:,:].reshape(nombre_d_elements,nx*ny*npixel)  
    
    instance.fit(tampon, data['y'])
    
    return instance
    
def donneMoiLaClasse(X, data, SklearnInstance, k):
    compteur = [0,0,0,0,0,0,0,0,0,0]
    nx, ny, npixel = X.shape
    voisin = SklearnInstance.kneighbors(X.reshape(1,nx*ny*npixel) , k, return_distance=False) 
    
    for i in range(len(voisin[0]-1)):
        compteur[(data['y'][voisin[0][i]][0]-1)%9] += 1
    
    return np.argmax(compteur)

def test(data, taille):
    bonneReponse = 0;
    SklearnInstance = getSklearnInstance(data)
    
    if taille == -1:
        taille = len(data['y'])
    
    for i in range(taille):
        if data['y'][i]%10 == donneMoiLaClasse(data["X"][:, :, :, i], data, SklearnInstance, 10):
            bonneReponse += 1
    
    print(bonneReponse)

if __name__ == "__main__":
    compteur = [0,0,0,0,0,0,0,0,0,0]
    train_data = loadmat('../train_32x32.mat')
    
    truc = NearestNeighbors()
    nx, ny, npixel, nombre_d_elements = train_data['X'][:,:,:,:].shape
    tampon = train_data['X'][:,:,:,:].reshape(nombre_d_elements,nx*ny*npixel)  
    
    truc.fit(tampon, train_data['y'])
    
    voisin = truc.kneighbors(train_data['X'][:,:,:,0].reshape(1,nx*ny*npixel) , 10, return_distance=False) 
    
    for i in range(len(voisin[0]-1)):
        compteur[(train_data['y'][voisin[0][i]][0]-1)%9] += 1
    
    solution = np.argmax(compteur)
    
    print(solution)
