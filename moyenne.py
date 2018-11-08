# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
import matplotlib.pyplot as plt;
import math as mt;
            
def calculerImageMoyenneClasseX(X,data):
    nbre = 0
    average = np.zeros((32,32,3),float)
    averageInt = np.zeros((32,32,3),int)
    
    for i in range(len(data['y'])):
        if data['y'][i] == X:
            nbre += 1
    
    for i in range(len(data['y'])):
        if data['y'][i] == X:
            average = average + data['X'][:,:,:,i]/nbre    

    for i in range(len(average)):
        for j in range(len(average[0])):
            for k in range(3):
                averageInt[i][j][k] = int(average[i][j][k])

    return averageInt
    
def test(trainData, testData):
    moyenne = []
    
    for i in range(1,10):
        moyenne.append(calculerImageMoyenneClasseX(i,trainData))  
        
    #image_idx = 0
    #print("Label:", trainData["y"][image_idx])
    #plt.imshow(trainData["X"][:, :, :, image_idx])
    #print(img)
    #plt.imshow(img)
    for i in range(100):
        if trainData['y'][i] == 1:
            print(estClasseXouY(trainData["X"][:, :, :, i], moyenne[0], moyenne[7]))
    
    #plt.show()
    
def estClasseXouY(image, moyenneX, moyenneY):
    resultat = 0  
    milieu = (moyenneX+moyenneY)/2
    vecteurNormal = moyenneX-moyenneY    
    
    tampon = image-milieu
    for i in range(32):
        for j in range(32):
            for k in range(3):
                resultat += vecteurNormal[i][j][k]*tampon[i][j][k]

    return resultat>0