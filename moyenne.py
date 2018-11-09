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
    bonneReponse = 0;    
    moyenne = []
    
    for i in range(1,10):
        moyenne.append(calculerImageMoyenneClasseX(i,trainData))  

    #image_idx = 0
    #print("Label:", trainData["y"][image_idx])
    #plt.imshow(moyenne[8])
    #plt.imshow(trainData['X'][:,:,:,0])
    #print(img)
    #plt.show()
    
    
    
    #for i in range(10):
    #    if trainData['y'][i] == donneMoiLaClasse(trainData["X"][:, :, :, i], moyenne):
    #        bonneReponse += 1
    
    print(bonneReponse)

def donneMoiLaClasse(image, moyenne):
    bonneClasse = 1
    classe = 2
    classeAtester = [3,4,5,6,7,8,9]

    while len(classeAtester) != 0:
        if(not estClasseXouY(image, moyenne[bonneClasse-1], moyenne[classe-1])):
            bonneClasse = classe
        classe = classeAtester.pop()
    
    return bonneClasse

# retourne true si c'est X et false sinon
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