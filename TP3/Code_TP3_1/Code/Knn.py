import distances #fonction de calcul de distance 
import evaluation #fonction pour la partie evaluation
import numpy as np

class Knn:

	def __init__(self,k,metrics = 'euclidean'):
		"""
		Initialiseur
		Args : 
			* k 		: nombre de plus proche voisins 	(int)
			* metrics 	: Le nom de la distance a utilisé 	(string)
				- euclidean 	: calcul la distance de euclidienne (valeur par défaut)
				- manhattan 	: calcul la distance de manhattan
				- cosine 		: calcul la distance cosine
				- hamming 		: calcul la distance de Hamming
				- jaccard 		: calcul la distance de Jaccard
		"""
		self.k = k
		self.metrics = metrics
			
	def train(self, train, train_labels):
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		"""
		# Récuperation des paramètres
		self.n = len(train)
		self.m = len(train[0])
		self.nbClasse = len(np.unique(train_labels))

		# Stockage des données
		self.train = train
		self.train_labels = train_labels.astype('float').astype('int')
		
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		# Initialisation des variables
		n = self.n
		distance = np.zeros((n))
		

		# Cacul de la matrice des distances
		if self.metrics == 'manhattan':
			p = 1
			for j in range(n):
				x2 = np.array(self.train[j])
				dist = distances.minkowski(p,x,x2)
				distance[j] = dist
		
		elif self.metrics == 'cosine':
			for j in range(n):
				x2 = np.array(self.train[j])
				dist = distances.cosine(x,x2)
				distance[j] = dist
		
		elif self.metrics == 'hamming':
			for j in range(n):
				x2 = np.array(self.train[j])
				dist = distances.hamming(x,x2)
				distance[j] = dist
		
		elif self.metrics == 'jaccard':
			for j in range(n):
				x2 = np.array(self.train[j])
				dist = distances.jaccard(x,x2)
				distance[j] = dist
		else:
			#Distance euclidienne par défaut
			p = 2
			for j in range(n):
				x2 = np.array(self.train[j])
				dist = distances.minkowski(p,x,x2)
				distance[j] = dist
		
		# Calcul des KPPV
		classe = np.zeros(self.nbClasse)
		k_idx_sorted = np.argpartition(distance, self.k+1)[:self.k]

		
		for idx in k_idx_sorted:
			classe[self.train_labels[idx]] +=1

		return np.argmax(classe)		
		

		
	def evaluate(self, X, y,affichage = True):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		"""
		y_test = y.astype('float').astype('int')

		# Calcul de la matrice de confusion
		confusionMatrix = np.zeros((self.nbClasse,self.nbClasse))
		for i,x in enumerate(X):
			pred = self.predict(x)
			confusionMatrix[pred][y_test[i]] +=1
		
		# Calcul des évaluations
		acc = []
		for classe in range(self.nbClasse):
			(A,P,R,F) = evaluation.evaluation(confusionMatrix,classe)
			if affichage:
				evaluation.affichage(confusionMatrix,A,P,R,F,classe)
			else:
				acc.append(A)
		return sum(acc)/self.nbClasse
		

		
		

