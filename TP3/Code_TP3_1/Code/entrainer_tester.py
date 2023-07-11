import numpy as np
import sys
import load_datasets
import NaiveBayes # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
import time # pour le temps d'execution

from sklearn.neighbors import KNeighborsClassifier  # Pour Q2.1.5
from sklearn.model_selection import KFold           # Pour Q2.1.5

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""
### Définition des fonctions ###
def irisK5():
    """
    Cette fonction effectue l'entrainement pour la base de données Iris
    """
    k=25
    # Initialisez/instanciez vos classifieurs avec leurs paramètres
    knnIris = Knn.Knn(k,metricsIris)
    # Entrainez votre classifieur
    knnIris.train(iris_train,iris_train_labels)
    return knnIris
    
def vihnoVerdeK5():
    """
    Cette fonction effectue l'entrainement pour la base de données VinhoVerde
    """
    k=1
    # Initialisez/instanciez vos classifieurs avec leurs paramètres
    knnVinhoVerde = Knn.Knn(k,metricsAbalones)
    # Entrainez votre classifieur
    knnVinhoVerde.train(vinhoVerde_train,vinhoVerde_train_labels)
    return knnVinhoVerde

def abalonesK5():
    """
    Cette fonction effectue l'entrainement pour la base de données Abalones
    """
    k=5
    # Initialisez/instanciez vos classifieurs avec leurs paramètres
    knnAbalones = Knn.Knn(k,metricsVinhoVerde)
    # Entrainez votre classifieur
    knnAbalones.train(abalones_train,abalones_train_labels)
    return knnAbalones

def evaluation(knn, dataset_X, dataset_y):
    """
    Cette fonction evalue l'entrainement sur un échantillons d'un jeu de donnée
    """
    knn.evaluate(dataset_X,dataset_y)

def KfoldIris():
    """
    Cette fonction répond à la question 2.1.4 pour la base de données Iris
    Retours:
        * best_k : Le k qui m'aximise l'exactitude sur le jeu de test
    """
    print("Jeux de donnée : IRIS")
    t1 = time.time()
    acc = []
    for k in K:
        accKfold = []
        for i in range(L):
            idxs = np.arange(len(iris_train)//L)+i*len(iris_train)//L
            X_train = np.delete(iris_train,idxs,0)
            y_train = np.delete(iris_train_labels,idxs,0)

            # Initialisez/instanciez vos classifieurs avec leurs paramètres
            knnIris = Knn.Knn(k,metricsIris)
            # Entrainez votre classifieur
            knnIris.train(X_train,y_train)
            accKfold.append(knnIris.evaluate(iris_test,iris_test_labels,False))
        avgAcc = sum(accKfold)/L
        acc.append(avgAcc)
        print(f"k = {k}, avg. acc. = {avgAcc}%")
    best_acc_idx = np.argmax(acc)
    best_k = increment*best_acc_idx+1
    print(f"Meileur k = {best_k} pour une exactitude de {round(acc[best_acc_idx]*100,2)}%")
    t2 = time.time()
    print(f"Temps d'execution = {round(t2-t1,2)} s")
    return best_k

def KfoldVihnoVerde():
    """
    Cette fonction répond à la question 2.1.4 pour la base de données VinhoVerde
    Retours:
        * best_k : Le k qui m'aximise l'exactitude sur le jeu de test
    """
    print("\n\n")
    print("Jeux de donnée : VINHOVERDE")
    t1 = time.time()
    acc = []
    for k in K:
        accKfold = []
        for i in range(L):
            idxs = np.arange(len(vinhoVerde_train)//L)+i*len(vinhoVerde_train)//L
            X_train = np.delete(vinhoVerde_train,idxs,0)
            y_train = np.delete(vinhoVerde_train_labels,idxs,0)
            # Initialisez/instanciez vos classifieurs avec leurs paramètres
            knnVinhoVerde = Knn.Knn(k,metricsVinhoVerde)
            # Entrainez votre classifieur
            knnVinhoVerde.train(X_train,y_train)
            accKfold.append(knnVinhoVerde.evaluate(vinhoVerde_test,vinhoVerde_test_labels,False))
            avgAcc = sum(accKfold)/L
        acc.append(avgAcc)
        print(f"k = {k}, avg. acc. = {avgAcc}%")
    best_acc_idx = np.argmax(acc)
    best_k = increment*best_acc_idx+1
    print(f"Meileur k = {best_k} pour une exactitude de {round(acc[best_acc_idx]*100,2)}%")
    t2 = time.time()
    print(f"Temps d'execution = {round(t2-t1,2)} s")
    return best_k

def KfoldAbalones():
    """
    Cette fonction répond à la question 2.1.4 pour la base de données Abalones
    Retours:
        * best_k : Le k qui m'aximise l'exactitude sur le jeu de test
    """
    print("\n\n")
    print("Jeux de donnée : ABALONES")
    t1 = time.time()
    acc = []
    for k in K:
        accKfold = []
        for i in range(L):
            idxs = np.arange(len(abalones_train)//L)+i*len(abalones_train)//L
            X_train = np.delete(abalones_train,idxs,0)
            y_train = np.delete(abalones_train_labels,idxs,0)
            # Initialisez/instanciez vos classifieurs avec leurs paramètres
            knnAbalones = Knn.Knn(k,metricsAbalones)
            # Entrainez votre classifieur
            knnAbalones.train(X_train,y_train)
            knnAbalones.evaluate(abalones_test,abalones_test_labels,False)
            avgAcc = sum(accKfold)/L
        acc.append(avgAcc)
        print(f"k = {k}, avg. acc. = {avgAcc}%")
    best_acc_idx = np.argmax(acc)
    best_k = increment*best_acc_idx+1
    print(f"Meileur k = {best_k} pour une exactitude de {round(acc[best_acc_idx]*100,2)}%")
    t2 = time.time()
    print(f"Temps d'execution = {round(t2-t1,2)} s")
    return best_k

def sklearnKnn(k, X_train, y_train, X_test, y_test):
    """
    Cette fonction entraine un model et retourne sont score pour un k donné
    Args :
        * k         : Nombre de plus proche voisin                  (int)
        * X_train   : Données d'entrainement                        (np.matrix)
        * y_train   : Labels d'entrainement                         (np.array)
        * X_test    : Données d'évaluation                          (np.matrix)
        * y_test    : Labels d'évaluation                           (np.array)
    Retours :
        score : retourne l'exactitude moyenne sur le deu de donnée d'évaluation donné
    """

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    return knn.score(X_test,y_test)

def sklearnKfold(L, K, X, y):
    """
    Cette fonction entraine un model et retourne sont score pour un k donné
    Args :
        * L             : Nombre de plis                                (int)
        * K             : Nombre de plus proche voisin                  (range)
        * metrics       : La fonction permetant de calculer la distance (str)
        * X             : Données d'entrainement                        (np.matrix)
        * y             : Labels d'entrainement                         (np.array)

    Retours :
        best_k          : Le meilleurs k trouver par validation croisée (int)
        best_acc        : La meilleure exactitude pour le best_k
    """
    # Initialisation
    kf = KFold(n_splits=L)
    acc = []

    # Validation croisé
    for k in K:
        acc_k = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            acc_k.append(sklearnKnn(k, X_train, y_train, X_test, y_test))
        acc.append(sum(acc_k)/L)
    
    best_k = 4*np.argmax(acc)+1
    best_acc = np.max(acc)
    return (best_k,best_acc)
    


# Initialisez vos paramètres
## Paramètre pour le Knn
metricsIris = 'euclidean'
metricsVinhoVerde = 'euclidean'
metricsAbalones = 'euclidean'
## Paramètre pour Bayes Naif
## Paramètre pour K-fold
L = 10
Kmin = 1
Kmax = 52
increment = 4
K = range(Kmin,Kmax,increment)

# Charger/lire les datasets'
train_ratio = 0.8 #80% pour train et 20% pour test

(iris_train, iris_train_labels, iris_test, iris_test_labels) = load_datasets.load_iris_dataset(train_ratio)
(vinhoVerde_train, vinhoVerde_train_labels, vinhoVerde_test, vinhoVerde_test_labels) = load_datasets.load_wine_dataset(train_ratio)
(abalones_train, abalones_train_labels, abalones_test, abalones_test_labels) = load_datasets.load_abalone_dataset(train_ratio)


# Entrainement
print("Entrainement")
#knnIris = irisK5()
#knnVinhoVerde = vihnoVerdeK5()
knnAbalones = abalonesK5()


"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
print("Évalutation sur les jeux d'entrainement")
print("\n\n")
print("Jeux de donnée : IRIS")
#evaluation(knnIris, iris_train, iris_train_labels)
print("\n\n")
print("Jeux de donnée : VINHOVERDE")
#evaluation(knnVinhoVerde, vinhoVerde_train, vinhoVerde_train_labels)
print("\n\n")
print("Jeux de donnée : ABALONES")
#evaluation(knnAbalones, abalones_train, abalones_train_labels)


# Tester votre classifieur

"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
print("Évalutation sur les jeux de test")
print("\n\n")
print("Jeux de donnée : IRIS")
#evaluation(knnIris, iris_test, iris_test_labels)
print("\n\n")
print("Jeux de donnée : VINHOVERDE")
#evaluation(knnVinhoVerde, vinhoVerde_test, vinhoVerde_test_labels)
print("\n\n")
print("Jeux de donnée : ABALONES")
#evaluation(knnAbalones, abalones_test, abalones_test_labels)

"""
Q 2.1.4
"""
#k_iris = KfoldIris() # Meileurs k pour iris : 25 (99.11%) temps d'execution = 8.23 s
#k_vihno = KfoldVihnoVerde() # Meileurs k pour vihno verde : 1 (87.93%) temps d'execution = 2572 s
#k_abalones = KfoldAbalones() # Meileurs k pour Abalones : Trop long

"""
Q2.1.5
"""

t1 = time.time()
(k_iris_sklearn, acc_iris) = sklearnKfold(L, K, iris_train, iris_train_labels) # Meileurs k pour iris :
t2 = time.time()
(k_vihno_sklearn, acc_vihno) = sklearnKfold(L, K, vinhoVerde_train, vinhoVerde_train_labels) # Meileurs k pour vihno verde :
t3 = time.time()


#print(f"IRIS : best_k = {k_iris_sklearn}, acc. = {round(acc_iris*100,2)}%, time = {round(t2-t1,2)} s")
#print(f"VIHNO : best_k = {k_vihno_sklearn}, acc. = {round(acc_vihno*100,2)}%, time = {round(t3-t2,2)} s")


