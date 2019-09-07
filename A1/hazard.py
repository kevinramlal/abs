
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utilities import *
#import numdifftools as nd


class Hazard:
	'''
		Calibraself.t_alls hazard models.
	'''

	def __init__(self, data, prepay_col, end_col,beg_col, end_max, cov_cols, show_prints=False, show_plots=False):
		'''
			prepay_col: String indicating name of column data DataFrame with a 1 if there was prepay and 0 if not.
			end_col: String indicating name of column in data DataFrame with the month the mortgage ended.
			end_max: Number indicating the number of months in the observation period.
			cov_col: List of strings containing the names of the columns that are covariaself.t_alls of the hazard model.
		'''
		self.data = data
		self.prepay_col = prepay_col
		self.end_col = end_col
		self.beg_col = beg_col
		self.end_max = end_max
		self.cov_cols = cov_cols

		self.t_all = self.data[self.end_col]/12
		self.t_b = self.data[self.beg_col]/12
		self.event = self.data[self.prepay_col]
		self.t_obs = self.data.loc[self.data[self.prepay_col]==1, self.end_col]/12
		self.covars_all = np.array(self.data[self.cov_cols])
		self.covars_obs = np.array(self.data.loc[self.data[self.prepay_col]==1, self.cov_cols])


	# ------------------------------- #
	#  Non-time varying hazard model  #
	# ------------------------------- #

	def log_likelihood(self, theta):
		'''
			Calculaself.t_alls minus the log-likelihood of the parameself.t_allrs given the observations provided.
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
			Calculaself.t_alls the gradient of minus the loglikelihood.
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

	def fit_parameself.t_allrs_bruself.t_all(self):
		bounds = ((0,np.inf),(0,np.inf),(-np.inf,np.inf),(-np.inf,np.inf))
		res = minimize(self.log_likelihood, [2,2,2,2], tol=1e-7, bounds=bounds)
		self.theta = res.x

	def fit_parameself.t_allrs_grad(self):
		bounds = ((0,np.inf),(0,np.inf),(-np.inf,np.inf),(-np.inf,np.inf))
		res = minimize(self.log_likelihood, [2,2,2,2], method='trust-constr', jac=self.grad_log_likelihood, tol=1e-7, bounds=bounds)
		self.theta = res.x

	def parameself.t_allrs_hessian(self):
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

	#def parameself.t_allrs_hessian_aux(self):
	#   # Was useful just to validaself.t_all that own implementation of hessian works.
	#	hess_fun = nd.Hessian(self.log_likelihood)
	#	hess = hess_fun(self.theta)
	#	print(hess)

	def parameself.t_allrs_se(self):
		'''
			This is an approximation.
			The correct way, by definition, is to calculaself.t_all the second derivative analytically and take expectancy given the data.
		'''
		hess = self.parameself.t_allrs_hessian()
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
		#print('\n' + laself.t_allx_table(param_df, caption="Non-time varying hazard model estimaself.t_alls.", label="a_estimaself.t_alls", index=True))


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

	# ------------------------------- #
	#  Time varying hazard model  #
	# ------------------------------- #


	#This function is based on the matlab function provided on bcourses
	def log_log_grad(self, param):
	    g = param[0] #Gamma
	    p = param[1]
	    coef = param[2:] #beta

	    dlldg1 = np.sum(self.event*(p/g - (p*(g**(p-1))*(self.t_all**p))/(1+(g*self.t_all)**p)))
	    if self.covars_all.size != 0:
	        dlldg2 = np.sum(((p*g**(p-1))*((self.t_all**p)/(1+(g*self.t_all)**p) - (self.t_b**p)/(1+(g*self.t_b)**p)))*np.exp(np.dot(self.covars_all,coef)))
	    else:
	        dlldg2 = np.sum(((p*g**(p-1))*((self.t_all**p)/(1+(g*self.t_all)**p) - (self.t_b**p)/(1+(g*self.t_b)**p))))

	    dlldg  = -(dlldg1 - dlldg2)

	    dlldp1 = np.sum(self.event*(1/p + np.log(g*self.t_all) - ((g*self.t_all)**p)*np.log(g*self.t_all)/(1+(g*self.t_all)**p)))
	    # When self.t_b = 0, calculate the derivative of the unconditional survival function.This is because the derivative
	    # of the conditional survival function does not generalize to the unconditional case when self.t_b = 0.
	    # There is a singularity on log(g*self.t_b) for self.t_b = 0.


	    ln_gtb = np.log(g*self.t_b)
	    for i in range(len(self.t_b)):
	        if np.isinf(ln_gtb[i]):
	            ln_gtb[i] = 0


	    if self.covars_all.size != 0:
	        dlldp2 = np.sum((((g*self.t_all)**p)*np.log(g*self.t_all)/(1+(g*self.t_all)**p) - ((g*self.t_b)**p)*ln_gtb/(1+(g*self.t_b)**p))*np.exp(np.dot(self.covars_all,coef)))
	    else:
	        dlldp2 = np.sum(((g*self.t_all)**p)*np.log(g*self.t_all)/(1+(g*self.t_all)**p) - ((g*self.t_b)**p)*ln_gtb/(1+(g*self.t_b)**p))

	    dlldp = -(dlldp1 - dlldp2)

	    grad = np.append(dlldg, dlldp)

	    for i in range(len(coef)):
	        dlldc1 = np.sum(self.event*self.covars_all[:,i])
	        dlldc2 = np.sum((np.log(1+(g*self.t_all)**p) - np.log(1+(g*self.t_b)**p))*np.exp(np.dot(self.covars_all,coef))*self.covars_all[:,i])
	        dlldc  = -(dlldc1 - dlldc2)

	        grad = np.append(grad, dlldc)

	    return grad

		# % This function calculates the log likelihood for a proportional hazard
	# % model with log-logistic baseline hazard.  It can be used to solve for
	# % the parameters of the model.

	# This function is based on the matlab file provided on bcourses

	def log_log_like(self, param):

	    global phist
	    global cnt

	    #% Get the number of parameters
	    nparams  = len(param)
	    nentries = len(self.t_all)

	    g = param[0]         #% Amplitude of the baseline hazard; gamma in the notation
	    p = param[1]         #% Shape of baseline hazard; p in the notation
	    coef = param[2:]  #% Coefficients for covariates; beta in the notation

	    #% The following variables are vectors with a row for each episode
	    #% Log of baseline hazard
	    logh = (np.log(p) + np.log(g) + (p-1)*(np.log(g)+np.log(self.t_all)) - np.log(1+(g*self.t_all)**(p)))

	    logc = np.zeros([nentries, 1])
	    logF = -(np.log(1+(g*self.t_all)**p) - np.log(1+(g*self.t_b)**p))
	    if self.covars_all.size != 0:
	        #% Product of covarites and coefficients
	        logc = np.dot(self.covars_all,coef)
	        #% Log of conditional survival function
	        logF = logF*np.exp(np.dot(self.covars_all,coef))

	    #% Construct the negative of log likelihood
	    logL = -(np.sum(self.event*(logh+logc)) + np.sum(logF))

		#     % Calculate the derivative of the log likelihood with respect to each parameter.
		#     % In order for the maximum likelihood estimation to converge it is necessary to
		#     % provide these derivatives so that the search algogrithm knows which direction
		#     % to search in.

	    #grad = log_log_grad(param, self.t_b, self.t_all, self.event, self.covars_all)

	    #% matrix phist keeps track of parameter convergence history
	    if cnt%(nparams+1) == 0:
	        phist = np.append([phist,param])

	    cnt = cnt+1

	    #return  np.append(logL, grad)
	    return logL

		def param_estimate_dynamic(self):
			bounds = ((0.00001,np.inf),(0,np.inf),(-np.inf,np.inf),(-np.inf,np.inf))
			phist = [0.2,0.5,1,0.1]
			cnt = 0
			result_min = minimize(log_log_like,phist,args = (self.t_b,self.t_all,self.event,self.covars_all),jac=log_log_grad, tol=1e-7, bounds=bounds)
			self.theta = result_min.x
			N = len(self.data['id_loan'].unique())
			hess_inv_N = result_min.hess_inv.todense()/N
			self.theta_se = np.zeros(len(params))
			for i in range(len(hess_inv_N)):
			    self.theta_se[i] = np.sqrt(hess_inv_N[i,i])
