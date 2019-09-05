
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Hazard:
	'''
		This class will calibrate hazard models.
	'''

	def __init__(self, data=None, prepay_col=None, end_col=None, end_max=None, cov_cols=None):
		'''
			prepay_col: String indicating name of column data DataFrame with a 1 if there was prepay and 0 if not.
			end_col: String indicating name of column in data DataFrame with the month the mortgage ended.
			end_max: Number indicating the number of months in the observation period.
			cov_col: List of strings containing the names of the columns that are covariates of the hazard model.
		'''
		self.data = data
		self.prepay_col = prepay_col
		self.end_col = end_col
		self.end_max = end_max
		self.cov_cols = cov_cols

		self.t_all = self.data[self.end_col]/12
		self.t_obs = self.data.loc[self.data[self.prepay_col]==1, self.end_col]/12
		self.covars_all = np.array(self.data[self.cov_cols])
		self.covars_obs = np.array(self.data.loc[self.data[self.prepay_col]==1, self.cov_cols])


	# ------------------------------- #
	#  Non-time varying hazard model  #
	# ------------------------------- #

	def log_likelihood(self, theta):
		'''
			Calculates minus the log-likelihood of the parameters given the observations provided.
		'''
		p = theta[0] # p in notation
		g = theta[1] # gamma in notation
		b = np.array(theta[2:]) # beta in notation

		log1 = np.sum(np.log(p) + np.log(g) + (p-1)*np.log(g*self.t_obs) - np.log(1+(g*self.t_obs)**p))
		log2 = np.sum(np.matmul(self.covars_obs, b))
		log3 = -np.sum(np.exp(np.matmul(self.covars_all, b))*np.log(1+(g*self.t_all)**p))
		logL = log1 + log2 + log3
		return -logL

	def grad_log_likelihood(self, theta):
		'''
			Calculates the gradient of minus the loglikelihood.
		'''
		p = theta[0] # p in notation
		g = theta[1] # gamma in notation
		b = np.array(theta[2:]) # beta in notation


		dlog_p_obs = np.sum(1/p + np.log(g*self.t_obs) - (g*self.t_obs)**p*np.log(g*self.t_obs)/(1+(g*self.t_obs)**p))
		dlog_p_all = -np.sum(np.exp(np.matmul(self.covars_all, b))*(g*self.t_all)**p*np.log(g*self.t_all)/(1+(g*self.t_all)**p))
		dlog_p = dlog_p_obs + dlog_p_all

		dlog_g_obs = np.sum(p/g - (self.t_obs**p*p*g**(p-1))/(1+(g*self.t_obs)**p))
		dlog_g_all = -np.sum(np.exp(np.matmul(self.covars_all, b))*self.t_all**p*p*g**(p-1)/(1+(g*self.t_all)**p))
		dlog_g = dlog_g_obs + dlog_g_all

		dlog_b_obs = np.sum(self.covars_obs, axis=0)
		dlog_b_all = np.zeros(2)
		for i in range(len(b)):
			dlog_b_all[i] = -np.sum(np.exp(np.matmul(self.covars_all, b))*np.log(1+(g*self.t_all)**p)*self.covars_all[:,i], axis=0)
		dlog_b = dlog_b_obs + dlog_b_all

		grad = [dlog_p, dlog_g] + list(dlog_b)
		return -np.array(grad)

	def fit_parameters_brute(self):
		bounds = ((0,np.inf),(0,np.inf),(-np.inf,np.inf),(-np.inf,np.inf))
		res = minimize(self.log_likelihood, [2,2,2,2], tol=1e-7, bounds=bounds)
		self.theta = res.x

	def fit_parameters_grad(self):
		bounds = ((0,np.inf),(0,np.inf),(-np.inf,np.inf),(-np.inf,np.inf))
		res = minimize(self.log_likelihood, [2,2,2,2], method='trust-constr', jac=self.grad_log_likelihood, tol=1e-7, bounds=bounds)
		self.theta = res.x

	def baseline_hazard(self):
		p = self.theta[0] # p in notation
		g = self.theta[1] # gamma in notation

		t = np.arange(0,self.end_max+1)/12
		b_lambda = g*p*(g*t)**(p-1)/(1+(g*t)**p)


		plt.plot(t, b_lambda)
		plt.xlabel('Years')
		plt.ylabel('Baseline hazard')
		plt.show()
		return b_lambda


