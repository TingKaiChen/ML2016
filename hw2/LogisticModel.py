import numpy as np
import sys

class LRmodel(object):
	"""docstring for LRmodel"""
	def __init__(self, eta=0., w=0., bias=0., iteration=10000, lam=0., 
		adagrad=0., train_in=0., train_out=0., test_in=0., test_out=0.):
		self.eta = eta
		self.w = w
		self.bias = bias
		self.iteration = iteration
		self.lam = lam
		self.adagrad = adagrad
		self.train_in = train_in
		self.train_out = train_out
		self.test_in = test_in
		self.test_out = test_out
		self.sigma_sq = np.zeros(self.w.shape)
		self.sigma_z_sq = 0
		self.accu = 0
	def run(self):
		### Gradient descent ###
		for it in range(self.iteration):
			z = self.bias+np.dot(self.train_in, self.w)
			est_out = sigmoidFunc(z)
			err = self.train_out-est_out
			self.accu = (self.train_out==(est_out>0.5)).sum()/float(len(self.train_out))

			# Update weightings
			pdv_w = (-1)*np.dot(err.T, self.train_in).T#+2*self.w*self.lam 	# Partial derivative of w1~n
			self.sigma_sq += pdv_w**2
			if self.adagrad == True:
				self.w = self.w-self.eta*pdv_w/np.sqrt(self.sigma_sq)
			else: 
				self.w = self.w-self.eta*pdv_w

			# Update the bias
			pdv_b = (-1)*err.sum()
			self.sigma_z_sq += pdv_b**2
			if self.adagrad == True:
				self.bias = self.bias-self.eta*pdv_b/np.sqrt(self.sigma_z_sq)
			else:
				self.bias = self.bias-self.eta*pdv_b

			# Print out
			if self.accu == np.inf:
				print
				print "iter: ", it
				break
			sys.stdout.write('\riter: '+str(it)+' Accurate: '+str(self.accu))
			sys.stdout.flush()
	def nf_test_result(self):
		"""Return a loss of n-fold testing data"""
		est_out = sigmoidFunc(self.bias+np.dot(self.test_in, self.w))
		test_accu = (self.test_out==(est_out>0.5)).sum()/float(len(self.test_out))
		return test_accu
	def print_nf_result(self):
		"""Print out the n-fold test result"""
		print ' test_accu: '+str(self.nf_test_result())
	def testing_output(self, testX_in):
		"""For real testing data output"""
		z = self.bias+np.dot(testX_in, self.w)
		return (sigmoidFunc(z)>0.5).astype(int)
	def getW(self):
		return self.w
	def getBias(self):
		return self.bias
	def getTrainLoss(self):
		return self.accu

def sigmoidFunc(x):
	return 1./(1.+np.exp(-x))

		