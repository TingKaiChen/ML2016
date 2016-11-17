import pickle
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.optimizers import Adadelta

### Parameters ###


### Load image ###
if '-l' not in sys.argv:
	all_label = pickle.load(open('./data/all_label.p','rb'))
	all_label = np.array(all_label)
	all_unlabel = pickle.load(open('./data/all_unlabel.p', 'rb'))
	all_unlabel = np.array(all_unlabel)
	test = pickle.load(open('./data/test.p', 'rb'))
	test = test['data']
	test = np.array(test)
	np.save('np_label', all_label)
	np.save('np_unlabel', all_unlabel)
	np.save('np_test', test)
else:
	all_label = np.load('np_label.npy')
	all_unlabel = np.load('np_unlabel.npy')
	test = np.load('np_test.npy')

### Data slicing ###
cv_train = np.zeros((5000,3,32,32))
for i in xrange(10):
	for j in xrange(500):
		cv_train[i*500+j, :, :, :] = all_label[i][j].reshape((3,32,32))
cv_train1 = np.concatenate((cv_train[0::3, :, :, :], cv_train[1::3, :, :, :]))
cv_train2 = np.concatenate((cv_train[1::3, :, :, :], cv_train[2::3, :, :, :]))
cv_train3 = np.concatenate((cv_train[2::3, :, :, :], cv_train[0::3, :, :, :]))

cv_out = np.ones((10,500))*[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
cv_out = cv_out.reshape((1,5000))
cv_out = np_utils.to_categorical(cv_out[0], 10)
cv_out1 = np.concatenate((cv_out[0::3, :], cv_out[1::3, :]))
cv_out2 = np.concatenate((cv_out[1::3, :], cv_out[2::3, :]))
cv_out3 = np.concatenate((cv_out[2::3, :], cv_out[0::3, :]))

cv_testin1 = cv_train[2::3, :, :, :]
cv_testin2 = cv_train[0::3, :, :, :]
cv_testin3 = cv_train[1::3, :, :, :]

cv_testout1 = cv_out[2::3, :]
cv_testout2 = cv_out[0::3, :]
cv_testout3 = cv_out[1::3, :]


### Architecture ###
if '-m' not in sys.argv:
	# model = Sequential()
	# model.add(Convolution2D(32,3,3,input_shape=(3,32,32), dim_ordering='th'))
	# model.add(Activation('relu'))
	# model.add(MaxPooling2D((2,2), dim_ordering='th'))
	# model.add(Activation('relu'))
	# model.add(Flatten())
	# model.add(Dense(output_dim=500))
	# model.add(Activation('relu'))
	# model.add(Dense(output_dim=10))
	# model.add(Activation('softmax'))

	model = Sequential()

	model.add(Convolution2D(32, 3, 3,
	                        input_shape=(3,32,32), dim_ordering='th'))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3, dim_ordering='th'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, dim_ordering='th'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, dim_ordering='th'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	adad = Adadelta(lr=0.3, rho=0.95)
	model.compile(optimizer=adad, loss='categorical_crossentropy', metrics=['accuracy'])

	### Train ###
	print "###### CV 1 ######"
	model.fit(cv_train1, cv_out1, batch_size=50, nb_epoch=40)
	score_cv1 = model.evaluate(cv_testin1, cv_testout1)
	print
	print "CV1. Loss:", score_cv1[0], "  Accuracy:", score_cv1[1]
	print "###### CV 2 ######"
	model.fit(cv_train2, cv_out2, batch_size=50, nb_epoch=40)
	score_cv2 = model.evaluate(cv_testin2, cv_testout2)
	print 
	print "CV2. Loss:", score_cv2[0], "  Accuracy:", score_cv2[1]
	print "###### CV 3 ######"
	model.fit(cv_train3, cv_out3, batch_size=50, nb_epoch=40)
	score_cv3 = model.evaluate(cv_testin3, cv_testout3)
	print 
	print "CV3. Loss:", score_cv3[0], "  Accuracy:", score_cv3[1]
	print 
	print "Average accuracy: ", (score_cv1[1]+score_cv2[1]+score_cv3[1])/3
	print
	print "###### All Data ######"
	model.fit(cv_train, cv_out, batch_size=50, nb_epoch=40)

	### Save model ###
	model.save('my_model.h5')
else:
	model = load_model('my_model.h5')

### Prediction ###
testdata = np.zeros((10000, 3, 32, 32))
for i in xrange(10000):
	testdata[i, :, :, :] = test[i].reshape((3, 32, 32))

testout = model.predict(testdata)
testout = np.argmax(testout, axis = 1)

with open('prediction.csv', 'w+') as pd:
	pd.write('ID,class\n')
	for i in xrange(10000):
		pd.write(str(i)+','+str(testout[i])+'\n')










