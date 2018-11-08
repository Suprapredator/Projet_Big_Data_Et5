# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
import math as mt;
            
def calculerImageMoyenneClasseX(X,data):
    nbre = 0
    
    for i in range(len(data['y'])):
        if data['y'][i] == X:
            nbre += 1
    
    ok = True
    for i in range(len(data['y'])):
        if data['y'][i] == X:
            if ok:
                average = data['X'][:,:,:,i]/nbre
                ok = False
            else:
                average = average + data['X'][:,:,:,i]/nbre
            #print(average[0][0])
    
    
    averageInt = data['X'][:,:,:,0]
    for i in range(len(average)):
        for j in range(len(average[0])):
            for k in range(3):
                averageInt[i][j][k] = int(average[i][j][k])

    return averageInt