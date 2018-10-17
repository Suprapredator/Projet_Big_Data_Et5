# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
import matplotlib.pyplot as plt;
#from scipy.io import loadmat;

def get_perpendicular_vector(v):
    n = np.array([1.0,1.0])    
    if v[1] != 0.:
        n[1] = -v[0]/v[1]
    else:
        n[0] = -v[1]/v[0]
    return n

def is_in_good_place(integer, point, milieu, vector_normal):
    res1 = point-milieu
    res2 = vector_normal[0]*res1[0]+vector_normal[1]*res1[1]
    if integer == 0:
        return res2>0    
    else:
        return res2<0

def get_y_for_graph(M, A):
    res = [0,0,0,0]
    if(A[1] == 0):
        res[0] = M[0]
        res[1] = M[0]
        res[2] = -5
        res[3] = 5
    else:
        res[0] = -5
        res[1] = 5
        res[2] = -5 * -A[0]/A[1] + (A[1]*M[1]+A[0]*M[0])/(A[1])
        res[3] = 5 * -A[0]/A[1] + (A[1]*M[1]+A[0]*M[0])/(A[1])
    return res
        
W1 = np.array([(1,1,2,2),(1,2,1,2)])
W2 = np.array([(0,1,2,3,3,3),(3,3,3,2,1,0)])

print('Donnees 1:')
print(W1)
print('Donnees 2:')
print(W2)

B1 = np.array([(np.average(W1[0])),(np.average(W1[1]))])
B2 = np.array([(np.average(W2[0])),(np.average(W2[1]))])

print('Bary 1:')
print(B1)
print('Bary 2:')
print(B2)

M = (B1+B2)/2

print('Milieu :')
print(M)

A = B1-B2

print('A :')
print(A)

print('Résultat W1:')
for i in range(0,W1[0].size):
    t = np.array([W1[0][i],W1[1][i]])
    print(is_in_good_place(0, t, M, A))

print('Résultat W2:')
for i in range(0,W2[0].size):
    t = np.array([W2[0][i],W2[1][i]])
    print(is_in_good_place(1, t, M, A))

# Points W1
plt.plot(W1[0], W1[1], 'ro')
# Points W2
plt.plot(W2[0], W2[1], 'g+')
# Droite
droite = get_y_for_graph(M, A)
plt.plot([droite[0], droite[1]], [droite[2], droite[3]], 'b-')

plt.axis([-5, 5, -5, 5])
plt.show()