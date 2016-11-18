import pickle
import sys
import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.utils import np_utils
from keras.optimizers import Adadelta
# from scipy.misc import toimage
from sklearn.neighbors import NearestNeighbors


### Load image ###
if '-l' not in sys.argv:
	all_label = pickle.load(open(sys.argv[1]+'all_label.p','rb'))
	all_label = np.array(all_label)
	all_unlabel = pickle.load(open(sys.argv[1]+'all_unlabel.p', 'rb'))
	all_unlabel = np.array(all_unlabel)
	test = pickle.load(open(sys.argv[1]+'test.p', 'rb'))
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
cv_train = np.zeros((5000,3072))
for i in xrange(10):
	for j in xrange(500):
		cv_train[i*500+j, :] = all_label[i][j]

cv_out = np.ones((10,500))*[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
cv_out = cv_out.reshape((1,5000))
cv_out = np_utils.to_categorical(cv_out[0], 10)

unlabelin = np.zeros((45000, 3072))
for i in xrange(45000):
	unlabelin[i, :] = all_unlabel[i]

autoData = np.concatenate((cv_train, unlabelin))/255.
autoData = autoData.reshape((50000, 3, 32, 32))

# toimage(autoData[0,:,:,:]*255).show()

if '-train' in sys.argv:
	### Autoencoder ###
	if '-a' not in sys.argv:
		# autoData = autoData.reshape((50000, 3072))
		# inputImg = Input(shape=(3072,))
		# encoded = Dense(128, activation='relu')(inputImg)
		# encoded = Dense(64, activation='relu')(encoded)
		# encoded = Dense(32, activation='relu')(encoded)

		# decoded = Dense(64, activation='relu')(encoded)
		# decoded = Dense(128, activation='relu')(decoded)
		# decoded = Dense(3072, activation='relu')(decoded)

		inputImg = Input(shape=(3,32,32))
		encoded = Convolution2D(16, 3, 3, dim_ordering='th', activation='relu', border_mode='same')(inputImg)
		encoded = MaxPooling2D((2, 2), dim_ordering='th', border_mode='same')(encoded)
		encoded = Convolution2D(8, 3, 3, dim_ordering='th', activation='relu', border_mode='same')(encoded)
		encoded = MaxPooling2D((2, 2), dim_ordering='th', border_mode='same')(encoded)
		encoded = Convolution2D(8, 3, 3, dim_ordering='th', activation='relu', border_mode='same')(encoded)
		encoded = MaxPooling2D((2, 2), dim_ordering='th', border_mode='same')(encoded)

		decoded = Convolution2D(8, 3, 3, dim_ordering='th', activation='relu', border_mode='same')(encoded)
		decoded = UpSampling2D((2, 2), dim_ordering='th')(decoded)
		decoded = Convolution2D(8, 3, 3, dim_ordering='th', activation='relu', border_mode='same')(decoded)
		decoded = UpSampling2D((2, 2), dim_ordering='th')(decoded)
		decoded = Convolution2D(16, 3, 3, dim_ordering='th', activation='relu', border_mode='same')(decoded)
		decoded = UpSampling2D((2, 2), dim_ordering='th')(decoded)
		decoded = Convolution2D(3, 3, 3, dim_ordering='th', activation='sigmoid', border_mode='same')(decoded)

		autoencoder = Model(input=inputImg, output=decoded)
		encoder = Model(input=inputImg, output=encoded)
		adad = Adadelta(lr=1, rho=0.95)
		autoencoder.compile(optimizer=adad, loss='binary_crossentropy')
		autoencoder.fit(autoData, autoData, nb_epoch=10, batch_size=5)

		# autoencoder.save('autoencoder_model.h5')
		# encoder.save('encoder_model.h5')

	### Feature data ###
	print 
	print "Extracting image features"
	allfeat = encoder.predict(autoData)
	labelfeat = allfeat[:5000,:,:,:].reshape(5000,128)
	unlabelfeat = allfeat[5000:,:,:,:].reshape(45000,128)

	### KNN ###
	print 
	print "##### KNN processing #####"
	nbrs = NearestNeighbors(n_neighbors=15, algorithm='auto').fit(labelfeat)
	print "find neighbors"
	distances, unlabelnbrs = nbrs.kneighbors(unlabelfeat)
	unlabelnbrs = (unlabelnbrs/500).astype(int)
	unlabelclass = []
	for nb in unlabelnbrs:
		counts = np.bincount(nb)
		unlabelclass.append(np.argmax(counts))
	unlabelclass = np_utils.to_categorical(unlabelclass, 10)

	### Combine data ###
	print "Combining all data"
	print 
	cv_out = np.concatenate((cv_out, unlabelclass))

	cv_train1 = np.concatenate((autoData[0::3, :, :, :], autoData[1::3, :, :, :]))
	cv_train2 = np.concatenate((autoData[1::3, :, :, :], autoData[2::3, :, :, :]))
	cv_train3 = np.concatenate((autoData[2::3, :, :, :], autoData[0::3, :, :, :]))

	cv_out1 = np.concatenate((cv_out[0::3, :], cv_out[1::3, :]))
	cv_out2 = np.concatenate((cv_out[1::3, :], cv_out[2::3, :]))
	cv_out3 = np.concatenate((cv_out[2::3, :], cv_out[0::3, :]))

	cv_testin1 = autoData[2::3, :, :, :]
	cv_testin2 = autoData[0::3, :, :, :]
	cv_testin3 = autoData[1::3, :, :, :]

	cv_testout1 = cv_out[2::3, :]
	cv_testout2 = cv_out[0::3, :]
	cv_testout3 = cv_out[1::3, :]

	### Architecture ###
	if '-m' not in sys.argv:
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

		adad = Adadelta(lr=1, rho=0.95)
		model.compile(optimizer=adad, loss='categorical_crossentropy', metrics=['accuracy'])

		### Train ###
		print "###### CV 1 ######"
		model.fit(cv_train1, cv_out1, batch_size=20, nb_epoch=5)
		score_cv1 = model.evaluate(cv_testin1, cv_testout1)
		print
		print "CV1. Loss:", score_cv1[0], "  Accuracy:", score_cv1[1]
		print "###### CV 2 ######"
		model.fit(cv_train2, cv_out2, batch_size=20, nb_epoch=5)
		score_cv2 = model.evaluate(cv_testin2, cv_testout2)
		print 
		print "CV2. Loss:", score_cv2[0], "  Accuracy:", score_cv2[1]
		print "###### CV 3 ######"
		model.fit(cv_train3, cv_out3, batch_size=20, nb_epoch=5)
		score_cv3 = model.evaluate(cv_testin3, cv_testout3)
		print 
		print "CV3. Loss:", score_cv3[0], "  Accuracy:", score_cv3[1]
		print 
		print "Average accuracy: ", (score_cv1[1]+score_cv2[1]+score_cv3[1])/3
		print
		print "###### All Data ######"
		model.fit(autoData, cv_out, batch_size=20, nb_epoch=5)

		model.save('trained_model')
	else:
		model = load_model('trained_model')

### Prediction ###
testdata = test.reshape((10000, 3, 32, 32))

testout = model.predict(testdata)
testout = np.argmax(testout, axis = 1)
np.save('testout', testout)

if '-test' in sys.argv:
	testout = np.load('testout.npy')
	with open(sys.argv[3], 'w+') as pd:
		pd.write('ID,class\n')
		for i in xrange(10000):
			pd.write(str(i)+','+str(testout[i])+'\n')
