import csv
import numpy as np
import sys

if len(sys.argv) == 3:
	### Parsing data ###
	data = np.genfromtxt(sys.argv[1], delimiter=',')
	data = np.delete(data, 0, 1)			# Delete first column
	data = np.nan_to_num(data)				# Replace nan with 0

	din = data[:, :-1]
	dout = data[:, -1][np.newaxis].T
	din_0 = din[dout.T[0] == 0]		# class 0 input
	din_1 = din[dout.T[0] == 1]		# class 1 input

	fNum = din.shape[1]				# # of features
	num_0 = din_0.shape[0]			# # of class 0
	num_1 = din_1.shape[0]			# # of class 1

	P_c0 = float(num_0)/(num_0+num_1)
	P_c1 = float(num_1)/(num_0+num_1)

	### Mean & covariance ###
	mean_0 = np.sum(din_0, axis=0)/num_0
	mean_0 = mean_0.reshape((fNum, 1))
	mean_1 = np.sum(din_1, axis=0)/num_1
	mean_1 = mean_1.reshape((fNum, 1))
	cov_0 = np.dot((din_0.T-mean_0), (din_0.T-mean_0).T)/float(num_0)
	cov_1 = np.dot((din_1.T-mean_1), (din_1.T-mean_1).T)/float(num_1)
	cov = P_c0*cov_0+P_c1*cov_1


	### Probability ###
	detCov = np.linalg.det(cov)**0.5
	invCov = np.linalg.inv(cov)
else:
	### Probability ###
	[invCov, detCov, mean_0, mean_1, fNum, P_c0, P_c1] = np.load('model.npy')

def P_xc0(x):
	z = -0.5*np.dot((x-mean_0).T, invCov)
	z = np.dot(z, (x-mean_0))
	return (1/(2*np.pi)**(fNum/2))*(1/detCov)*np.exp(z)
def P_xc1(x):
	z = -0.5*np.dot((x-mean_1).T, invCov)
	z = np.dot(z, (x-mean_1))
	return (1/(2*np.pi)**(fNum/2))*(1/detCov)*np.exp(z)

def P_c1x(x):
	if (P_xc0(x)*P_c0+P_xc1(x)*P_c1)==0:
		return 0	# Default class = 1
	else:
		result = P_xc1(x)*P_c1/(P_xc0(x)*P_c0+P_xc1(x)*P_c1)
		return result
if len(sys.argv) == 3:
	out = np.zeros(dout.shape)
	for i in xrange(out.shape[0]):
		out[i] = P_c1x(din[i,:][np.newaxis].T)
	out = (out>0.5).astype(int)
	Accu = (out==dout).sum()/float(dout.shape[0])
	print Accu
	np.save('model', [invCov, detCov, mean_0, mean_1, fNum, P_c0, P_c1])
else:
	### Testing data ###
	testdata = np.genfromtxt(sys.argv[2], delimiter=',')
	testdata = np.delete(testdata, 0, 1)	# Delete first column
	testdata = np.nan_to_num(testdata)			# Replace nan with 0

	out = np.zeros((testdata.shape[0],1))
	for i in xrange(testdata.shape[0]):
		out[i] = P_c1x(testdata[i,:][np.newaxis].T)
	out = (out>0.5).astype(int)


	filename = sys.argv[3]
	with open(filename, 'w+') as outfile:
		outfile.write('id,label\n')
		for i in range(len(out)):
			outfile.write(str(i+1)+','+str(out[i][0])+'\n')



