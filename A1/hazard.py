
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utilities import *
#import numdifftools as nd
pd.set_option('display.max_columns', 15)

class Hazard:
	'''
		Calibrates hazard models.
	'''

	def __init__(self, data, prepay_col, end_col, end_max, cov_cols, show_prints=False, show_plots=False):
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
		self.show_prints = show_prints
		self.show_plots = show_plots

		self.t_all = self.data[self.end_col]
		self.t_obs = self.data.loc[self.data[self.prepay_col]==1, self.end_col]
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

	def parameters_hessian(self):
		eps = 1e-4
		k = len(self.theta)
		hess = np.zeros((k,k))
		for i in range(k):
			theta_up = np.copy(self.theta)
			theta_dn = np.copy(self.theta)
			theta_up[i] = theta_up[i] + eps
			theta_dn[i] = theta_dn[i] - eps
			grad_up = self.grad_log_likelihood(theta_up)
			grad_dn = self.grad_log_likelihood(theta_dn)
			hess[i] = (grad_up-grad_dn)/(2*eps)
		return hess

	#def parameters_hessian_aux(self):
	#   # Was useful just to validate that own implementation of hessian works.
	#	hess_fun = nd.Hessian(self.log_likelihood)
	#	hess = hess_fun(self.theta)
	#	print(hess)

	def parameters_se(self):
		'''
			This is an approximation. 
			The correct way, by definition, is to calculate the second derivative analytically and take expectancy given the data.
		'''
		hess = self.parameters_hessian()
		var = np.diagonal(np.linalg.inv(hess)) # We already flipped the sign in the log-likelihood.
		n = self.data.shape[0]
		se = np.sqrt(var)/np.sqrt(n)

		param_df = pd.DataFrame(self.theta, columns=['Value'])
		param_df['Std. Error'] = se
		param_names = ['p', 'gamma']
		for i in range(len(self.theta)-2):
			param_names += ['beta_'+str(i+1)]
		param_df.index = param_names
		print('\nPart a:\n\n' + str(param_df))
		#print('\n' + latex_table(param_df, caption="Non-time varying hazard model estimates.", label="a_estimates", index=True))


	def baseline_hazard(self, t):
		p = self.theta[0] # p in notation
		g = self.theta[1] # gamma in notation

		base_hz = g*p*(g*t)**(p-1)/(1+(g*t)**p)

		if self.show_plots:
			plt.plot(t, base_hz)
			plt.xlabel('Years')
			plt.ylabel('Baseline hazard')
			plt.show()

		return base_hz

	def calculate_prepayment(self, t, covars):
		base_hz = self.baseline_hazard(t)
		b = self.theta[2:]
		prepayment = base_hz*np.exp(np.matmul(covars, b))
		return prepayment
