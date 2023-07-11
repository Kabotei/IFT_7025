import numpy as np
import sys
import time
import load_datasets
import DecisionTree # importer la classe de l'arbre de décision
import NeuralNet# importer la classe du Knn
import matplotlib.pyplot as plt
#importer d'autres fichiers et classes si vous en avez développés


# Pour la question 1 e)
from sklearn import tree

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""


### Définition des fonctions ###
def irisTree(iris_train, iris_train_labels):
    """
    Cette fonction effectue l'entrainement pour la base de données Iris avec
    l'arbre de décision
    """
    # Initialisez/instanciez vos classifieurs avec leurs paramètres
    treeIris = DecisionTree.DecisionTree()
    # Entrainez votre classifieur
    treeIris.train(iris_train, iris_train_labels)
    return treeIris


def vihnoVerdeTree():
    """
    Cette fonction effectue l'entrainement pour la base de données VinhoVerde
    avec l'arbre de décision
    """
    # Initialisez/instanciez vos classifieurs avec leurs paramètres
    treeVinhoVerde = DecisionTree.DecisionTree()
    # Entrainez votre classifieur
    treeVinhoVerde.train(vinhoVerde_train, vinhoVerde_train_labels)
    return treeVinhoVerde


def abalonesTree():
    """
    Cette fonction effectue l'entrainement pour la base de données Abalones
    avec l'arbre de décision
    """
    # Initialisez/instanciez vos classifieurs avec leurs paramètres
    treeAbalones = DecisionTree.DecisionTree()
    # Entrainez votre classifieur
    treeAbalones.train(abalones_train, abalones_train_labels)
    return treeAbalones

def evaluation(clf, dataset_X, dataset_y):
    """
    Cette fonction evalue l'entrainement sur un échantillon d'un jeu de donnée
    """
    clf.evaluate(dataset_X, dataset_y)

def learningCurve():
    y = []
    ratios = [(i+1) for i in range(99)]
    for train_ratio in ratios:
        for i in range(1):
            #Init acc
            acc = []

            #Spliting
            (iris_train, iris_train_labels, iris_test, iris_test_labels) = load_datasets.load_iris_dataset(train_ratio/100)

            #Training
            tree = irisTree(iris_train, iris_train_labels)

            #Evaluating
            acc.append(tree.evaluate(iris_test,iris_test_labels, affichage=False))
        y.append(sum(acc)/len(acc))
    
    #Affichage
    plt.figure()
    plt.plot(ratios,y)
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.show()

# def KfoldIris():
#     """
#     Cette fonction répond à la question 2.1.4 pour la base de données Iris
#     Retours:
#         * best_k : Le k qui m'aximise l'exactitude sur le jeu de test
#     """
#     print("Jeux de donnée : IRIS")
#     t1 = time.time()
#     acc = []
#     for k in K:
#         accKfold = []
#         for i in range(L):
#             idxs = np.arange(len(iris_train) // L) + i * len(iris_train) // L
#             X_train = np.delete(iris_train, idxs, 0)
#             y_train = np.delete(iris_train_labels, idxs, 0)
#
#             # Initialisez/instanciez vos classifieurs avec leurs paramètres
#             knnIris = Knn.Knn(k, metricsIris)
#             # Entrainez votre classifieur
#             knnIris.train(X_train, y_train)
#             accKfold.append(knnIris.evaluate(iris_test, iris_test_labels, False))
#         avgAcc = sum(accKfold) / L
#         acc.append(avgAcc)
#         print(f"k = {k}, avg. acc. = {avgAcc}%")
#     best_acc_idx = np.argmax(acc)
#     best_k = increment * best_acc_idx + 1
#     print(f"Meilleur k = {best_k} pour une exactitude de {round(acc[best_acc_idx] * 100, 2)}%")
#     t2 = time.time()
#     print(f"Temps d'execution = {round(t2 - t1, 2)} s")
#     return best_k


def irisNN():
    """
    Cette fonction effectue l'entrainement pour la base de données Iris avec le réseau de neurones
    """
    # Initialisez/instanciez vos classifieurs avec leurs paramètres
    nnIris = NeuralNet.NeuralNet()
    # Entrainez votre classifieur
    nnIris.train(iris_train, iris_train_labels)
    return nnIris


def vihnoVerdeNN():
    """
    Cette fonction effectue l'entrainement pour la base de données VinhoVerde avec le réseau de neurones
    """
    # Initialisez/instanciez vos classifieurs avec leurs paramètres
    nnVinhoVerde = NeuralNet.NeuralNet()
    # Entrainez votre classifieur
    nnVinhoVerde.train(vinhoVerde_train, vinhoVerde_train_labels)
    return nnVinhoVerde


def abalonesNN():
    """
    Cette fonction effectue l'entrainement pour la base de données Abalones avec le réseau de neurones
    """
    # Initialisez/instanciez vos classifieurs avec leurs paramètres
    nnAbalones = NeuralNet.NeuralNet()
    # Entrainez votre classifieur
    nnAbalones.train(abalones_train, abalones_train_labels)
    return nnAbalones


def sklearnTree(critere, X_train, y_train, X_test, y_test):
    """
    Cette fonction entraine un arbre de décision à l'aide de la librairie scikit-learn et
    un critère de division et le score en test
    Args :
        * critere   : Critère de division (entropy ou gini)         (str)
        * X_train   : Données d'entrainement                        (np.matrix)
        * y_train   : Labels d'entrainement                         (np.array)
        * X_test    : Données d'évaluation                          (np.matrix)
        * y_test    : Labels d'évaluation                           (np.array)
    Retours :
        score : retourne l'exactitude moyenne sur le jeu de donnée test
    """

    treeSK = tree.DecisionTreeClassifier(criterion=critere)
    treeSK.fit(X_train, y_train)
    return treeSK.score(X_test, y_test)


# def sklearnKfold(L, K, X, y):
#     """
#     Cette fonction entraine un model et retourne sont score pour un k donné
#     Args :
#         * L             : Nombre de plis                                (int)
#         * K             : Nombre de plus proche voisin                  (range)
#         * metrics       : La fonction permetant de calculer la distance (str)
#         * X             : Données d'entrainement                        (np.matrix)
#         * y             : Labels d'entrainement                         (np.array)
#
#     Retours :
#         best_k          : Le meilleurs k trouver par validation croisée (int)
#         best_acc        : La meilleure exactitude pour le best_k
#     """
#     # Initialisation
#     kf = KFold(n_splits=L)
#     acc = []
#
#     # Validation croisé
#     for k in K:
#         acc_k = []
#         for train_index, test_index in kf.split(X):
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#             acc_k.append(sklearnKnn(k, X_train, y_train, X_test, y_test))
#         acc.append(sum(acc_k) / L)
#
#     best_k = 4 * np.argmax(acc) + 1
#     best_acc = np.max(acc)
#     return (best_k, best_acc)
#
#
# def sklearnNBAbalones(X_train, y_train, X_test, y_test):
#     X_train = pd.DataFrame(X_train)
#     X_test = pd.DataFrame(X_test)
#
#     enc = LabelEncoder()
#     X_train.iloc[:, 0] = enc.fit_transform(X_train.iloc[:, 0])
#     X_test.iloc[:, 0] = enc.fit_transform(X_test.iloc[:, 0])
#
#     enc1 = LabelEncoder()
#     y_train = enc1.fit_transform(y_train)
#     y_test = enc1.fit_transform(y_test)
#
#     clf = GaussianNB()
#     clf.fit(X_train, y_train)
#
#     pred = clf.predict(X_test)
#     return sum(y_test == pred) / len(pred)


# Initialisation des paramètres
## Paramètres pour le Knn

## Paramètres pour K-fold


# Charger/lire les datasets'
train_ratio = 0.7  # 70% pour train et 30% pour test est recommandé

(iris_train, iris_train_labels, iris_test, iris_test_labels) = load_datasets.load_iris_dataset(train_ratio)
(vinhoVerde_train, vinhoVerde_train_labels, vinhoVerde_test, vinhoVerde_test_labels) = load_datasets.load_wine_dataset(
    train_ratio)
(abalones_train, abalones_train_labels, abalones_test, abalones_test_labels) = load_datasets.load_abalone_dataset(
    train_ratio)

print("\n")
print("##################################################################")
print("Entrainement des modèles")
print("##################################################################")
print("\n")

print("Instanciation des arbres de décision et entrainement...")
treeIris = irisTree(iris_train, iris_train_labels)
treeVinhoVerde = vihnoVerdeTree()
treeAbalones = abalonesTree()
print("Les arbres de décisions sont instanciés et entrainés.")
#
# # start = time.time()
# print("Instanciation des réseaux de neurones et entrainement...")
# nnIris = irisNN()
# nnVinhoVerde = vihnoVerdeNN()
# nnAbalones = abalonesNN()
# print("Les réseau de neurones sont instanciés et entrainés.")

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
print("\n")
print("##################################################################")
print("Évaluation sur les jeux d'entrainement")
print("##################################################################")
print("\n")

print("Jeu de données en entrainement : IRIS avec le classifieur : DecisionTree")
print('------------------------------------')
evaluation(treeIris, iris_train, iris_train_labels)
# print("Jeu de données en entrainement : IRIS avec le classifieur : NeuralNet")
# print('------------------------------------')
# evaluation(nnIris, iris_train, iris_train_labels)
print("\n")
print("Jeu de données en entrainement : VINHOVERDE avec le classifieur : DecisionTree")
print('------------------------------------')
evaluation(treeVinhoVerde, vinhoVerde_train, vinhoVerde_train_labels)
# print("Jeu de données en entrainement : VINHOVERDE avec le classifieur : NeuralNet")
# print('------------------------------------')
# evaluation(nnVinhoVerde, vinhoVerde_train, vinhoVerde_train_labels)
print("\n")
print("Jeu de données en entrainement : ABALONES avec le classifieur : DecisionTree")
print('------------------------------------')
evaluation(treeAbalones, abalones_train, abalones_train_labels)
# print("Jeu de données en entrainement : ABALONES avec le classifieur : NeuralNet")
# print('------------------------------------')
# evaluation(nnAbalones, abalones_train, abalones_train_labels)

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
print("\n\n")
print("##################################################################")
print("Évaluation sur les jeux de test")
print("##################################################################")
print("\n")

print("Jeu de données en test : IRIS avec le classifieur : DecisionTree")
print('------------------------------------')
evaluation(treeIris, iris_test, iris_test_labels)
# print("Jeu de données en test : IRIS avec le classifieur : NeuralNet")
# print('------------------------------------')
# evaluation(nnIris, iris_test, iris_test_labels)
print("\n")
print("Jeu de données en test : VINHOVERDE avec le classifieur : DecisionTree")
print('------------------------------------')
evaluation(treeVinhoVerde, vinhoVerde_test, vinhoVerde_test_labels)
# print("Jeu de données en test : VINHOVERDE avec le classifieur : NeuralNet")
# print('------------------------------------')
# evaluation(nnVinhoVerde, vinhoVerde_test, vinhoVerde_test_labels)
print("\n")
print("Jeu de données en test : ABALONES avec le classifieur : DecisionTree")
print('------------------------------------')
evaluation(treeAbalones, abalones_test, abalones_test_labels)
# print("Jeu de données en entrainement : ABALONES avec le classifieur : NeuralNet")
# print('------------------------------------')
# evaluation(nnAbalones, abalones_test, abalones_test_labels)


print("\n\n")
print("##################################################################")
print("Question 1 e)")
print("Entrainement d'un arbre de décision à l'aide de la librairie scikit-Learn ")
print("##################################################################")


t1 = time.time()
acc_iris_sklearn = sklearnTree("entropy", iris_train, iris_train_labels, iris_test, iris_test_labels)
t2 = time.time()
acc_vinhoverde_sklearn = sklearnTree("entropy", vinhoVerde_train, vinhoVerde_train_labels, vinhoVerde_test, vinhoVerde_test_labels)
t3 = time.time()
acc_abalones_sklearn = sklearnTree("entropy", abalones_train, abalones_train_labels, abalones_test, abalones_test_labels)
t4 = time.time()
#
print(f"IRIS : acc. = {round(acc_iris_sklearn * 100, 2)}%, temps d'exécution = {round(t2 - t1, 2)} s")
print(f"VINHOVERDE : acc. = {round(acc_vinhoverde_sklearn * 100, 2)}%, temps d'exécution = {round(t3 - t2, 2)} s")
print(f"ABALONES : acc. = {round(acc_abalones_sklearn * 100, 2)}%, temps d'exécution = {round(t4 - t3, 2)} s")

learningCurve()