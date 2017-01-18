
from func import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
'''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.advanced_activations import ELU

import tensorflow
from tensorflow.python.ops import control_flow_ops 
tensorflow.python.control_flow_ops = control_flow_ops
'''

print 'Reading Data'
train_file = 'train'
test_file = 'test.in'
output = 'rf1.csv'

X, y, dim_1, dim_2, dim_3, maptype = read_train(train_file)


X_test = read_test(test_file, dim_1, dim_2, dim_3)

#pca = PCA(n_components=30)

#X = np.concatenate((X_train, X_test))
#X = pca.fit_transform(X)

#X_train, y_train = X[:3526869], y[:3526869]
#X_valid, y_valid = X[3526869:], y[3526869:]

#X = X[:, :40]
#X_test = X_test[:, :40]

print 'Training'
#X = SelectKBest(chi2, k=35).fit_transform(X, y)
#clf = GradientBoostingClassifier(n_estimators=25, learning_rate=1.0, random_state=0)
clf = RandomForestClassifier(n_estimators=25, criterion='gini', max_features=20, warm_start=True)

scores = cross_val_score(clf, X, y, cv=5)
print scores
print np.mean(scores)

clf.fit(X, y)
print clf.score(X, y)
#print clf.score(X_valid, y_valid)

print 'Predicting'
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)
for i in xrange(len(pred)):
	if prob[i,2] != 0.0:
		pred[i] = 2
write_pred(output, pred)

'''
batch_size = 1024
nb_epoch = 20

# the data, shuffled and split between train and test sets


# convert class vectors to binary class matrices
Y = np_utils.to_categorical(y, len(types))

model = Sequential()
model.add(Dense(512, input_shape=(dim,)))
model.add(ELU())
#model.add(Dropout(0.2))
model.add(Dense(512))
model.add(ELU())
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(len(types)))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.1), metrics=['accuracy'])

history = model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2)
'''
