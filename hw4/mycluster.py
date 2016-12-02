import sys
# import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

class MyCluster(object):
	"""Different clustering models"""
	def __init__(self, featmat, ctype):
		self.featmat = featmat
		self.classmat = None
		self.ctype = ctype
		self.model = None
	def build(self, load=None):
		"""Build the clustering model with the feature matrix"""
		fp = sys.stdout
		if load != 'load':
			print 'Clustering feature vector...',
			fp.flush()
			fn = None
			if self.ctype == 'kmeans':
				self.model = KMeans(n_clusters=20, random_state=0)
				self.model.fit(self.featmat)
				fn = 'km1'
			elif self.ctype == 'lsa':
				svd = TruncatedSVD(n_components=20)
				normalizer = Normalizer(copy=False)
				lsa = make_pipeline(svd, normalizer)
				self.featmat = lsa.fit_transform(self.featmat)
				self.model = KMeans(n_clusters=80, random_state=0)
				self.model.fit(self.featmat)
				fn = 'lsa'
			print 'done'
			print 'Save model...',
			fp.flush()
			with open(fn,'wb') as fh:
				joblib.dump(self.model, fh)
			print 'done'
		else:
			print 'Load model...',
			fp.flush()
			fn = None
			if self.ctype == 'kmeans':
				fn = 'km1'
			elif self.ctype == 'lsa':
				fn = 'lsa'
			with open(fn,'rb') as fh:
				self.model = joblib.load(fh)
			print 'done'
	def predict(self, pairs):
		"""Predict whether the pair is in the same cluster"""
		predList = []
		self.classmat = self.model.predict(self.featmat)
		for i in xrange(len(pairs)):
			label1 = self.classmat[pairs[i][0]]
			label2 = self.classmat[pairs[i][1]]
			if label1 != label2:
				predList.append(0)
			else:
				predList.append(1)
			sys.stdout.write('\rPredicting... '+str(float(i)/len(pairs)*100)+'%')
			sys.stdout.flush()
		print '\rPredicting...done'
		return predList


		