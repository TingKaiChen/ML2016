from FeatureExtract import FeatureExtract
from mycluster import MyCluster
from sklearn.cluster import KMeans
import numpy as np 
import sys

if len(sys.argv) == 1:
	path = 'data/'
else:
	path = sys.argv[1]

# Feature extraction: BoW, TF-IDF
FE = FeatureExtract(ftype='TF-IDF')
FE.loadFile(path+'title_StackOverflow.txt')
featMat = FE.getFeature()
# Normalization

# Cluster
model = MyCluster(featMat, 'lsa')
model.build()

# Prediction
checklist = np.genfromtxt(path+'check_index.csv', 
	delimiter=',')
checklist = checklist[1:, 1:].astype(int)
result = model.predict(checklist)
with open(sys.argv[2], 'w') as f:
	f.write('ID,Ans\n')
	for i in xrange(len(checklist)):
		f.write(str(i)+','+str(result[i])+'\n')