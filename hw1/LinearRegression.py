import csv
import numpy as np
import sys
from LRmodel import LRmodel

### Parsing data ###
data = np.genfromtxt('train.csv', delimiter=',')
data = np.delete(data, 0, 0)			# Delete the 1st row
data = np.delete(data, [0, 1, 2], 1)	# Delete first 3 columns
data = np.nan_to_num(data)				# Replace nan with 0

### Slice out the input and output by 10 hrs ###
mat_in_tmp = data[0:18, 0:24]
for i in range(1, 240):
	mat_in_tmp = np.concatenate((mat_in_tmp, data[i*18:i*18+18, 0:24]), axis=1)
train_in = mat_in_tmp[0:18, 0:9].reshape((1, 162))
for mon in range(12):
	for i in range(471):
		if mon == 0 and i == 0:
			continue
		else:
			data_vec = mat_in_tmp[0:18, 480*mon+i: 480*mon+i+9].reshape((1, 162))
			train_in = np.concatenate((train_in, data_vec), axis=0)
train_out = mat_in_tmp[9, 9:480][np.newaxis].T
for mon in range(1, 12):
	out_vec = mat_in_tmp[9, 480*mon+9:480*mon+480][np.newaxis].T
	train_out = np.concatenate((train_out, out_vec), axis=0)

# multiple order
# train_in = np.concatenate((train_in, train_in**2, train_in**3, train_in**4), axis=1)

# # Feature scaling
# mean = np.sum(train_in, axis=0)/train_in.shape[0]
# sd = (np.sum((train_in-mean)**2, axis=0)/train_in.shape[0])**0.5
# train_in = (train_in-mean)/sd

### N-fold cross validation ###
train_in_1 = np.concatenate((train_in[0::3, :], train_in[1::3, :]))
train_in_2 = np.concatenate((train_in[1::3, :], train_in[2::3, :]))
train_in_3 = np.concatenate((train_in[0::3, :], train_in[2::3, :]))
train_out_1 = np.concatenate((train_out[0::3, 0], train_out[1::3, 0]))[np.newaxis].T
train_out_2 = np.concatenate((train_out[1::3, 0], train_out[2::3, 0]))[np.newaxis].T
train_out_3 = np.concatenate((train_out[0::3, 0], train_out[2::3, 0]))[np.newaxis].T
nf_test_in_1 = train_in[2::3, :]
nf_test_in_2 = train_in[0::3, :]
nf_test_in_3 = train_in[1::3, :]
nf_test_out_1 = train_out[2::3, 0][np.newaxis].T
nf_test_out_2 = train_out[0::3, 0][np.newaxis].T
nf_test_out_3 = train_out[1::3, 0][np.newaxis].T

### Model parameter ###
# eta = 0.00000000296		# Learning rate for pure LR
eta = 0.7		# Learning rate
w1 = np.zeros((162, 1))	# weightings
w2 = np.zeros((162, 1))
w3 = np.zeros((162, 1))
w = np.zeros((162, 1))
bias = 0
iteration = 100000		# Iteration times
lam = 0
useAdagrad = True

# multiple order
# w1 = np.concatenate((w1, w1, w1, w1), axis=0)
# w2 = np.concatenate((w2, w2, w2, w2), axis=0)
# w3 = np.concatenate((w3, w3, w3, w3), axis=0)
# w = np.concatenate((w, w, w, w), axis=0)


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

avgTrainLoss = (model1_f1.getTrainLoss()+model1_f2.getTrainLoss()
	+model1_f3.getTrainLoss())/3
avgLoss = (model1_f1.nf_test_result()+model1_f2.nf_test_result()+
	model1_f3.nf_test_result())/3
print ("test: "+str(avgLoss)+' train: '+str(avgTrainLoss)+
	" Learning rate: "+str(eta)+' lam: '+str(lam))

# Run with all training data
model1_all = LRmodel(eta, w, bias, iteration, lam, useAdagrad,
	train_in, train_out, 0, 0)
model1_all.run()
print 

### Parsing testing data ###
testdata = np.genfromtxt('test_X.csv', delimiter=',')
testdata = np.delete(testdata, [0, 1], 1)	# Delete first 2 columns
testdata = np.nan_to_num(testdata)			# Replace nan with 0

### Slice out the input ###
test_in = testdata[0:18, 0:9].reshape((1, 162))
for i in range(1, 240):
	test_vec = testdata[i*18:i*18+18, 0:9].reshape((1, 162))
	test_in = np.concatenate((test_in, test_vec), axis=0)

# # Feature scaling
# test_in = (test_in-mean)/sd

# multiple order
# test_in = np.concatenate((test_in, test_in**2, test_in**3, test_in**4), axis=1)

### Testing result ###
test_out = model1_all.testing_output(test_in)

# Output file
with open('linear_regression.csv', 'w+') as outfile:
	outfile.write('id,value\n')
	for i in range(240):
		outfile.write('id_'+str(i)+','+str(test_out[i, 0])+'\n')

# Parameter log
with open('parameter.log', 'a') as plog:
	plog.write('test: '+str(avgLoss)+' train: '+str(avgTrainLoss)+
		' iter: '+str(iteration)+' eta: '+str(eta)+' lam: '+
		str(lam)+'\n')




