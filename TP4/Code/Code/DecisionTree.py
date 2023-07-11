import evaluation #fonction pour la partie evaluation
import Arbre
import numpy as np
import random
"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

class DecisionTree: #nom de la class à changer

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations

		racine : noeud racine de l'arbre.  Lorsqu'instancié pour la première fois, la racine = None.
		"""
		self.racine = Arbre.Noeud()
		
	def train(self, train, train_labels,pruining = False): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		# Récupération des paramètres
		self.n = train.shape[0]
		self.p = train.shape[1]

		# Stockage des données et des informations concernant les classes
		# self.train_labels = train_labels.astype('float').astype('int')
		self.train_labels = train_labels.astype('float')
		self.train_labels = self.train_labels.reshape((self.n, 1))
		self.trainXy = np.concatenate((train, np.asarray(self.train_labels)), axis=1)
		self.classes = np.unique(self.train_labels)
		self.nbClasses = len(self.classes)

		# Entropie S (pour tous les attributs)
		entropies = np.zeros((self.nbClasses, 1))
		for i in range(self.nbClasses):
			prob = np.count_nonzero(train_labels == self.classes[i])/self.n
			entropies[i,0] = -prob * np.log2(prob)
		self.entropie_S = np.sum(entropies)

		self.construction_arbre(self.racine, attributs=list(range(self.p)), exemples=self.trainXy, exemples_parent=None, pruining = pruining)

		# print("L'arbre est entrainé.")
		# print("Voici l'arbre :")
		# self.imprimer_arbre(self.racine, "                 ")

	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""

		# On commence par la racine de l'arbre
		noeud_courant = self.racine
		pred = self.parcours_arbre(noeud_courant, x)
		return pred

	def evaluate(self, X, y, affichage=True):
		"""
		c'est la méthode qui va évaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		y_test = y.astype('float').astype('int')

		# Calcul de la matrice de confusion
		classes = np.unique(y)
		nbClasses = max(len(classes),self.nbClasses)
		confusionMatrix = np.zeros((nbClasses, nbClasses))
		for i, x in enumerate(X):
			pred = self.predict(x)
			confusionMatrix[pred][y_test[i]] += 1

		# Calcul des évaluations
		acc = []
		for idx_classe in range(nbClasses):
			(A, P, R, F) = evaluation.evaluation(confusionMatrix, idx_classe)
			if affichage:
				evaluation.affichage(confusionMatrix, A, P, R, F, idx_classe)
			else:
				acc.append(A)
		return sum(acc) / nbClasses

	def gain_info(self, idx_p, exemples):
		"""
		c'est la méthode qui calcule le gain d'information pour un attribut
		idx_p : l'index de l'attribut pour lequel on recherche la valeur d'entropie
		"""
		valeurs = np.unique(list(exemples[:,idx_p]))

		# Si l'attribut n'a qu'une seule valeur
		if len(valeurs) == 1 :
			valeur_meilleure_coupe = valeurs[0]
			entropies = np.zeros((self.nbClasses, 1))
			for c in range(self.nbClasses):
				prob = np.count_nonzero(exemples[:, -1] == self.classes[c]) / exemples.shape[0]
				entropies[c, 0] = -prob * np.log2(prob) if prob else 0
			entropie= np.sum(entropies)
			gain_Sv = self.entropie_S - entropie

		elif len(valeurs) == 2:
			valeur_meilleure_coupe = valeurs[0]

			valeurs_inf = np.where(exemples[:, idx_p] <= valeur_meilleure_coupe)[0]
			entropies_inf = np.zeros((self.nbClasses, 1))
			for c in range(self.nbClasses):
				prob = np.count_nonzero(exemples[valeurs_inf, -1] == self.classes[c]) / len(valeurs_inf)
				entropies_inf[c, 0] = -prob * np.log2(prob) if prob else 0
			entropie_inf = np.sum(entropies_inf)

			valeurs_sup = np.where(exemples[:, idx_p] > valeur_meilleure_coupe)[0]
			entropies_sup = np.zeros((self.nbClasses, 1))
			for c in range(self.nbClasses):
				prob = np.count_nonzero(exemples[valeurs_sup, -1] == self.classes[c]) / len(valeurs_sup)
				entropies_sup[c, 0] = -prob * np.log2(prob) if prob else 0
			entropie_sup = np.sum(entropies_sup)

			gain_Sv = self.entropie_S - (len(valeurs_inf)/exemples.shape[0] * entropie_inf
														+ len(valeurs_sup)/exemples.shape[0] * entropie_sup)

		else :
			gain_valeur = np.zeros((len(valeurs)-2,1))

			for i in range(1,len(valeurs)-1): # On commence à la 2e index et on fini à l'avant-dernière
				# Une catégorie est formée par les valeurs inférieures à la valeur
				valeurs_inf = np.where(exemples[:,idx_p] <= valeurs[i])[0]
				entropies_inf = np.zeros((self.nbClasses, 1))
				for c in range(self.nbClasses):
					prob = np.count_nonzero(exemples[valeurs_inf, -1] == self.classes[c])/len(valeurs_inf)
					entropies_inf[c,0] = -prob*np.log2(prob) if prob else 0
				entropie_inf = np.sum(entropies_inf)

				# Une catégorie est formée par les valeurs supérieures à la valeur
				valeurs_sup = np.where(exemples[:,idx_p] > valeurs[i])[0]
				entropies_sup = np.zeros((self.nbClasses, 1))
				for c in range(self.nbClasses):
					prob = np.count_nonzero(exemples[valeurs_sup,-1] == self.classes[c])/len(valeurs_sup)
					entropies_sup[c,0] = -prob*np.log2(prob) if prob else 0
				entropie_sup = np.sum(entropies_sup)

				gain_valeur[i-1, 0] = self.entropie_S - (len(valeurs_inf)/exemples.shape[0] * entropie_inf
														+ len(valeurs_sup)/exemples.shape[0] * entropie_sup)

			meilleure_coupe = np.argmax(gain_valeur[:,0])
			if meilleure_coupe.shape != ():
				idx_meilleure_coupe = random.randint(0, len(meilleure_coupe))
			else :
				idx_meilleure_coupe = meilleure_coupe
			valeur_meilleure_coupe = valeurs[idx_meilleure_coupe+1]   # +1 parce que la première valeur n'avait pas été considérée
			gain_Sv = gain_valeur[idx_meilleure_coupe,0]

		return (gain_Sv, valeur_meilleure_coupe)

	def construction_arbre(self, racine=None, attributs=None, exemples=None, exemples_parent=None, pruining = False, seuilElagage = 0.1):

		
		noeud_courant = racine

		# S'il ne reste aucun exemple, on retourne la classe la plus fréquente parmi les exemples parents

		if exemples.shape[0] == 0:
			nb_classe_i = np.zeros((self.nbClasses, 1))
			for i in range(self.nbClasses):
				nb_classe_i[i, 0] = np.count_nonzero(exemples_parent[:,-1] == self.classes[i])
			noeud_courant.decision = self.classes[np.argmax(nb_classe_i, 0)[0]]

		# Si tous les exemples sont de la même classification, alors
		elif len(np.unique(exemples[:,-1])) == 1:
			noeud_courant.decision = exemples[0, -1]

		# Si liste_attribut est vide, on retourne la classe la plus fréquente parmi les exemples
		elif (attributs != None) and len(attributs) == 0:
			nb_classe_i = np.zeros((self.nbClasses, 1))
			for i in range(self.nbClasses):
				nb_classe_i[i, 0] = np.count_nonzero(exemples[:,-1] == self.classes[i])
			noeud_courant.decision = self.classes[np.argmax(nb_classe_i[:,0])]

		# Sinon, on divise l'arbre selon l'attribut avec le plus grand gain d'information
		else :
			liste_attributs = attributs[:]

			# Choix de l'attribut pour division
			gain_info_par_attribut = np.zeros((len(liste_attributs), 1))
			valeur_coupe_par_attribut = np.zeros((len(liste_attributs), 1))
			for i in range(len(liste_attributs)) :
				gain_info_par_attribut[i,0], valeur_coupe_par_attribut[i,0] = self.gain_info(liste_attributs[i], exemples)
			idx_attribut_choisi = np.argmax(gain_info_par_attribut)
			attribut_choisi = liste_attributs[idx_attribut_choisi]

			# La règle de décision pour ce noeud correspond au gain d'information maximal
			noeud_courant.regle = (attribut_choisi, valeur_coupe_par_attribut[idx_attribut_choisi])

			# Le noeud devient la racine pour un sous-arbre
			liste_attributs_enfants = [a for a in liste_attributs if a != attribut_choisi]

			# Noeud gauche
			noeud_courant.gauche = Arbre.Noeud()
			exemples_gauche = exemples[np.where(exemples[:,attribut_choisi] <= valeur_coupe_par_attribut[idx_attribut_choisi])[0],:]
			self.construction_arbre(racine = noeud_courant.gauche, attributs=liste_attributs_enfants, exemples=exemples_gauche, exemples_parent=exemples)

			# Noeud droit
			noeud_courant.droit = Arbre.Noeud()
			exemples_droit = exemples[np.where(exemples[:,attribut_choisi] > valeur_coupe_par_attribut[idx_attribut_choisi])[0],:]
			self.construction_arbre(racine = noeud_courant.droit, attributs=liste_attributs_enfants, exemples=exemples_droit, exemples_parent=exemples)

	def parcours_arbre(self, racine, x):

		noeud_courant = racine

		if noeud_courant.decision is not None :
			pred = noeud_courant.decision

		else :
			attribut_coupe, valeur_coupe = noeud_courant.regle
			if x[attribut_coupe] <= valeur_coupe :
				pred = self.parcours_arbre(noeud_courant.gauche, x)
			else :
				pred = self.parcours_arbre(noeud_courant.droit, x)

		return pred.astype('int')

	def imprimer_arbre(self, noeud, espaces):
		noeud = noeud
		if noeud.decision is not None:
			print(noeud.decision)
			return
		else :
			attribut, valeur = noeud.regle
			print(espaces, "Attribut ", attribut, ' <= ou > que ', valeur)
			espaces = espaces[:-2]
			while noeud.decision is None :
				print('Noeuds gauches :')
				print(self.imprimer_arbre(noeud.gauche, espaces))
				print('Noeuds droits :')
				print(self.imprimer_arbre(noeud.droit, espaces))
				return

