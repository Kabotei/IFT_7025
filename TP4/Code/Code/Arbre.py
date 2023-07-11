"""
Cette classe servira à la construction d'un arbre de décision pour le classifieur
DecisionTree.
"""

import numpy as np

class Noeud:
	def __init__(self, **kwargs):
		"""
		Initialise un arbre.
		"""
		self.decision = None
		self.regle = None
		self.gauche = None
		self.droit = None


