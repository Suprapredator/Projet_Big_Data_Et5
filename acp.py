import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;
import math as mt;
import copy
from sklearn.decomposition import PCA
import time
import moyenne as moy
from sklearn.preprocessing import StandardScaler

def reductionDimension(data):
	

	ligne, colonne, rgb, index = data['X'][:, :, :, :].shape
	images = data['X'][:, :, :, :].reshape(index, ligne * colonne * rgb)
	
	scaler = StandardScaler(copy=False)
	
	debut = time.time()
	print("scaling...")
	images = scaler.fit_transform(images)
	print("%s sec" % (time.time() - debut))
	
	testPCA = PCA(n_components=10, copy=False)

	debut = time.time()
	print("fit+reduction...")
	reduc = testPCA.fit_transform(images)
	print("%s sec" % (time.time() - debut))

	"""print("resultat = ")
	print(images[0, :])"""

	print("copie des donnees")
	d2_data = copy.deepcopy(data)
	d2_data['X'] = images

	return d2_data

def getMoyennes2d(trainData):
	moyenne = []
	
	for i in range(1,11):
		moyenne.append(calculerImage2dMoyenneClasseX(i,trainData))
	
	return moyenne

def calculerImage2dMoyenneClasseX(X,data):
	listeImage = []
	
	for i in range(len(data['y'])):
		if data['y'][i] == X:
			listeImage.append(data['X'][i, :])

	return np.mean(listeImage, axis=0).astype(int)

def donneMoiLaClasse(image, moyenne):
	distance = mt.inf 
	
	for i in range(len(moyenne)):
		newDistance = np.linalg.norm(image-moyenne[i])
		if distance > newDistance:
			bonneClasse = i+1
			distance = newDistance
	
	return bonneClasse

def test(moyenne, data, taille):
	start_time = time.time()
	bonneReponse = 0;
	
	print("test...")
	if taille == -1 or taille > len(data['y']):
		taille = len(data['y'])
	
	for i in range(taille):
		if data['y'][i] == donneMoiLaClasse(data["X"][i, :], moyenne):
			bonneReponse += 1
	
	print('Résultat par moyenne: '+str(bonneReponse)+'/'+str(taille))
	print("--- %s seconds ---" % (time.time() - start_time)) 

if __name__ == "__main__":

	print("initialisation matrices...")
	train_data = loadmat('../train_32x32.mat')
	test_data = loadmat('../test_32x32.mat')
	perf_data = loadmat('../perfect_train_data.mat')
	ptest_data = loadmat('../perfect_test_data.mat')

	d2_data = reductionDimension(perf_data)
	moyennes = getMoyennes2d(d2_data)
	test(moyennes, d2_data, -1)

	"""image_idx = 0;
	print('Label:', train_data['y'][image_idx])
	plt.imshow(train_data['X'][:, :, :, image_idx])
	plt.show()"""
