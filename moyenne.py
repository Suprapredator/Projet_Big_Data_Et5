# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
import matplotlib.pyplot as plt;
import math as mt;
            
def calculerImageMoyenneClasseX(X,data):
    listeImage = []
    
    for i in range(len(data['y'])):
        if data['y'][i] == X:
            listeImage.append(data['X'][:,:,:,i])

    return np.mean(listeImage, axis=0).astype(int)
    
def test(moyenne, data, taille):
    bonneReponse = 0;
    
    if taille == -1:
        taille = len(data['y'])
    
    for i in range(taille):
        if data['y'][i] == donneMoiLaClasse(data["X"][:, :, :, i], moyenne):
            bonneReponse += 1
    
    print(bonneReponse)

def getMoyennes(trainData):
    moyenne = []
    
    for i in range(1,11):
        moyenne.append(calculerImageMoyenneClasseX(i,trainData))
    
    return moyenne

def donneMoiLaClasse(image, moyenne):
    distance = mt.inf 
    
    for i in range(len(moyenne)):
        newDistance = np.linalg.norm(image-moyenne[i])
        if distance > newDistance:
            bonneClasse = i+1
            distance = newDistance
    
    return bonneClasse