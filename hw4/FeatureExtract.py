import numpy as np 
import re
import sys
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtract(object):
	"""Extract features from input file"""
	def __init__(self, ftype):
		self.data = []
		self.featMat = None
		self.wordlist = []
		self.ftype = ftype
	def loadFile(self, filename):
		"""Load data from txt file"""
		with open(filename, 'r') as f:
			lines = f.readlines()
			if self.ftype == 'BoW':
				for line in lines:
					# Remove special sign "?+(),=
					line = re.sub(r'[\?\:\"\(\)\+\,\=]', ' ', line)
					line = line.lower()
					npline = np.array(line.split())
					self.data.append(npline)
			elif self.ftype == 'TF-IDF':
				for line in lines:
					self.data.append(line)
			elif self.ftype == 'word2vec':
				for line in lines:
					# Remove special sign ...;{}"?+(),=
					line = re.sub(r'[\...\;\{\}\?\:\"\(\)\+\,\=]', '', line)
					line = line.replace('<br>', '')
					line = line.replace('<br />', '')
					line = line.lower().split()
					# stop_words = [];
					# line = [w for w in line if w not in stop_words]
					if len(line) == 0:
						continue
					self.data.append(line)
	def getData(self):
		"""Return the raw data matrix"""
		return self.data
	def getModel(self, loadmodel):
		"""Return the word2vec model"""
		if loadmodel == False:
			print 'Building word2vec model... ',
			model = gensim.models.Word2Vec(self.data)
			print 'done.'
			print 'Saving word2vec model... ',
			model.save('w2v')
			print 'done.'
		else:
			print 'Loading word2vec model... ',
			model = gensim.models.Word2Vec.load('w2v')
			print 'done.'
		return model
	def getFeature(self):
		"""Return the feature array/matrix"""
		fp = sys.stdout
		print 'Extracting features...',
		fp.flush()
		if self.ftype == 'BoW':
			self.bow()
		elif self.ftype == 'TF-IDF':
			self.tf_idf()
		print 'done'
		return self.featMat
	def bow(self):
		# Create a word list of all data
		self.wordlist = []
		for i in xrange(len(self.data)):
			for j in xrange(len(self.data[i])):
				if self.data[i][j] not in wordlist:
					wordlist.append(self.data[i][j])
		# Specify each title with a BoW vector
		self.featMat = np.zeros((len(self.data), len(wordlist)))
		for tID in xrange(len(self.data)):
			for word in self.data[tID]:
				word = word
				self.featMat[tID][wordlist.index(word)] += 1
	def tf_idf(self):
		vectorizer = TfidfVectorizer(stop_words='english', 
			min_df=2, max_df=0.5)
		self.featMat = vectorizer.fit_transform(self.data)


if __name__ == '__main__':
	fe = FeatureExtract('TF-IDF')
	fe.loadFile('data/title_StackOverflow.txt')
	a = fe.getFeature()
	b = fe.getData()
	print a.shape
	print
	print a[0, 4280]
	print
	print b[0]


