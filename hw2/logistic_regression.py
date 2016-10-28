import csv
import numpy as np
import sys
from LogisticModel import LRmodel
import warnings
warnings.filterwarnings("ignore")

cn1 = 44
cn = 51
if len(sys.argv) == 3:
	# for cn1 in range(2,56):
	# 	for cn in range(cn1+1,57):
	### Parsing data ###
	data = np.genfromtxt(sys.argv[1], delimiter=',')
	data = np.delete(data, 0, 1)			# Delete first column
	data = np.nan_to_num(data)				# Replace nan with 0

	# Overfitting
	# train_in = train_in[:276, :]
	# train_out = train_out[:276, 0][np.newaxis].T

	# multiple order
	data = np.concatenate((data[:, :-1], (data[:, cn1]**3*data[:, cn]**3)[np.newaxis].T, data[:, -1][np.newaxis].T), axis=1)

	# # Feature scaling
	# mean = np.sum(train_in, axis=0)/train_in.shape[0]
	# sd = (np.sum((train_in-mean)**2, axis=0)/train_in.shape[0])**0.5
	# train_in = (train_in-mean)/sd

	### N-fold cross validation ###
	train_in_1 = np.concatenate((data[0::3, :-1], data[1::3, :-1]))
	train_in_2 = np.concatenate((data[1::3, :-1], data[2::3, :-1]))
	train_in_3 = np.concatenate((data[0::3, :-1], data[2::3, :-1]))
	train_out_1 = np.concatenate((data[0::3, -1], data[1::3, -1]))[np.newaxis].T
	train_out_2 = np.concatenate((data[1::3, -1], data[2::3, -1]))[np.newaxis].T
	train_out_3 = np.concatenate((data[0::3, -1], data[2::3, -1]))[np.newaxis].T
	nf_test_in_1 = data[2::3, :-1]
	nf_test_in_2 = data[0::3, :-1]
	nf_test_in_3 = data[1::3, :-1]
	nf_test_out_1 = data[2::3, -1][np.newaxis].T
	nf_test_out_2 = data[0::3, -1][np.newaxis].T
	nf_test_out_3 = data[1::3, -1][np.newaxis].T
	train_in = data[:, :-1]
	train_out = data[:, -1][np.newaxis].T

	### Model parameter ###
	# eta = 0.00000000009		# Learning rate for pure LR
	eta = 0.2		# Learning rate
	fNum = 58
	w1 = np.ones((fNum, 1))	# weightings
	w2 = np.ones((fNum, 1))
	w3 = np.ones((fNum, 1))
	w = np.ones((fNum, 1))
	bias = 0
	iteration = 10000		# Iteration times
	lam = 0
	useAdagrad = True

	# multiple order
	# w1 = np.concatenate((w1, w1), axis=0)
	# w2 = np.concatenate((w2, w2), axis=0)
	# w3 = np.concatenate((w3, w3), axis=0)
	# w = np.concatenate((w, w), axis=0)


	### Build model ###
	model1_f1 = LRmodel(eta, w1, bias, iteration, lam, useAdagrad,
		train_in_1, train_out_1, nf_test_in_1, nf_test_out_1)
	model1_f2 = LRmodel(eta, w2, bias, iteration, lam, useAdagrad,
		train_in_2, train_out_2, nf_test_in_2, nf_test_out_2)
	model1_f3 = LRmodel(eta, w3, bias, iteration, lam, useAdagrad,
		train_in_3, train_out_3, nf_test_in_3, nf_test_out_3)

	model1_f1.run()
	model1_f1.print_nf_result()
	model1_f2.run()
	model1_f2.print_nf_result()
	model1_f3.run()
	model1_f3.print_nf_result()

	avgTrainAccu = (model1_f1.getTrainLoss()+model1_f2.getTrainLoss()
		+model1_f3.getTrainLoss())/3
	avgAccu = (model1_f1.nf_test_result()+model1_f2.nf_test_result()+
		model1_f3.nf_test_result())/3
	# if avgAccu < 0.9256:
	# 	continue
	print ("test: "+str(avgAccu)+' train: '+str(avgTrainAccu)+
		" Learning rate: "+str(eta)+' lam: '+str(lam))

	# Run with all training data
	model1_all = LRmodel(eta, w, bias, iteration, lam, useAdagrad,
		train_in, train_out, 0, 0)
	model1_all.run()
	print 

	# Parameter log
	with open('parameter.log', 'a') as plog:
		plog.write('# '+str(cn1)+'*'+str(cn)+'\n')
		plog.write('test: '+str(avgAccu)+' train: '+str(avgTrainAccu)+
			' iter: '+str(iteration)+' eta: '+str(eta)+' lam: '+
			str(lam)+'\n')
	# Save model
	np.save('model', [model1_all.getW(), model1_all.getBias()])

if len(sys.argv) == 4:
	### Parsing testing data ###
	testdata = np.genfromtxt(sys.argv[2], delimiter=',')
	testdata = np.delete(testdata, 0, 1)	# Delete first column
	testdata = np.nan_to_num(testdata)			# Replace nan with 0

	# # Feature scaling
	# test_in = (test_in-mean)/sd

	# multiple order
	testdata = np.concatenate((testdata, (testdata[:, cn1]*testdata[:, cn])[np.newaxis].T), axis=1)

	### Testing result ###
	[test_w, test_b] = np.load('model.npy')
	model_test = LRmodel(w=test_w, bias=test_b)
	test_out = model_test.testing_output(testdata)

	# Output file
	filename = sys.argv[3]
	with open(filename, 'w+') as outfile:
		outfile.write('id,label\n')
		for i in range(len(test_out)):
			outfile.write(str(i+1)+','+str(test_out[i, 0])+'\n')

	




