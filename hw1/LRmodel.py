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
		self.trainLoss = 0
	def run(self):
		### Gradient descent ###
		for it in range(self.iteration):
			err = self.train_out-(self.bias+np.dot(self.train_in, self.w))
			self.trainLoss = ((np.square(err).sum())/len(self.train_out))**0.5
			
			# Update weightings
			pdv_w = (-2)*np.dot(err.T, self.train_in).T+2*self.w*self.lam 	# Partial derivative of w1~n
			self.sigma_sq += pdv_w**2
			if self.adagrad == True:
				self.w = self.w-self.eta*pdv_w/np.sqrt(self.sigma_sq)
			else: 
				self.w = self.w-self.eta*pdv_w

			# Update the bias
			pdv_b = (-2)*err.sum()
			self.sigma_z_sq += pdv_b**2
			if self.adagrad == True:
				self.bias = self.bias-self.eta*pdv_b/np.sqrt(self.sigma_z_sq)
			else:
				self.bias = self.bias-self.eta*pdv_b

			# Print out
			if self.trainLoss == np.inf:
				print
				print "iter: ", it
				break
			sys.stdout.write('\riter: '+str(it)+' Loss: '+str(self.trainLoss))
			sys.stdout.flush()
	def nf_test_result(self):
		"""Return a loss of n-fold testing data"""
		err = self.test_out-(self.bias+np.dot(self.test_in, self.w))
		return ((np.square(err).sum())/len(self.test_out))**0.5
	def print_nf_result(self):
		"""Print out the n-fold test result"""
		print ' nf_loss: '+str(self.nf_test_result())
	def testing_output(self, testX_in):
		"""For real testing data output"""
		return self.bias+np.dot(testX_in, self.w)
	def getW(self):
		return self.w
	def getBias(self):
		return self.bias
	def getTrainLoss(self):
		return self.trainLoss

		