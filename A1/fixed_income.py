
import pandas as pd
import numpy as np
from scipy.integrate import quad

class FixedIncome:
	'''
		This class contains all functions and algorithms needed to estimate fixed income models
	'''

	def __init__(self):
		pass


	# ----------------------- #
	#  Day count conventions  #
	# ----------------------- #

	def get_days_act_360(self, start_date, end_date):
	    return (end_date-start_date).days/360
	    
	def get_days_30I_360(self, start_date, end_date):
	    Y1 = start_date.year
	    Y2 = end_date.year
	    M1 = start_date.month
	    M2 = end_date.month
	    D1 = start_date.day
	    D2 = end_date.day
	    if D1==31:
	        D1=30
	    if D2==31:
	        D2=30
	    return (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1))/360

	# ---------------- #
	#  Hull and White  #
	# ---------------- #


	def hull_white_theta(self, t, dt, theta):
		'''
			Returns linear interpolation for theta(t) based on discrete values.
			theta contains discrete values of the theta function every dt years.
			If theta contains monthly values, then dt = 1/12.
			t is the value to interpolate theta from.
		'''
		index = t/dt
		index_1 = int(np.floor(index))
		index_2 = int(np.ceil(index))

		if index_1 == index_2:
			return theta[index_1]

		theta_1 = theta[index_1]
		theta_2 = theta[index_2]
		t_1 = index_1*dt
		t_2 = index_2*dt
		theta_t = (theta_2 - theta_1)/(t_2 - t_1)*(t - t_1) + theta_1
		return theta_t

	def hull_white_B(self, t, T, kappa):
		'''
			Calculates B(t,T) in Hull and White
		'''
		B = (1.0 - np.exp(-kappa * (T-t)))/kappa
		return B

	def hull_white_A_B(self, t, T, theta, kappa, sigma):
		'''
			Calculates A(t,T) in Hull and White
		'''

		B = self.hull_white_B(t, T, kappa)
		def hull_white_integrand(tau, dt, theta, kappa):
			return self.hull_white_B(tau, T, kappa)*self.hull_white_theta(tau, dt, theta)

		dt = 1/12
		I = quad(hull_white_integrand, t, T, args=(dt, theta, kappa))[0]
		A = -I + (sigma**2 / (2.0*kappa**2))*(T - t + (1.0 - np.exp(-2.0*kappa*(T - t)))/(2.0*kappa) - 2.0*B)
		return (A, B)

	def hull_white_instantaneous_spot_rate(self, t, T, Z, theta, kappa, sigma):
		'''
			Given a discount factor Z for the [t, T] period, it returns the implicit instantaneous spot rate
		'''
		AB =  self.hull_white_A_B(t, T, theta, kappa, sigma)
		A = AB[0]
		B = AB[1]
		r0 = -(np.log(Z)-A)/B
		return r0

	def hull_white_simulate_rates(self, n, r0, theta, kappa, sigma):
		'''
			Simulates n paths of instantaneous rates using the Hull and White model.
			dr(t) = (θ(t) − κr(t))dt + σdW(t)
		'''

		np.random.seed(0)
		r = np.zeros((n, len(theta)))
		dt = 1/12
		r[:, 0] = r0

		for i in range(1, len(theta)):
			w = np.random.normal(0, sigma, n)
			dr = (theta[i-1] - kappa*r[:, i-1])*dt + sigma*w
			r[:, i] = r[:, i-1] + dr

		return r

	def hull_white_discount_factor(self, r, t, T, theta, kappa, sigma):
		'''
			Returns discount factor from t to T.
			r is the instantaneous rate at t. It can be a numpy array.
		'''
		AB =  self.hull_white_A_B(t, T, theta, kappa, sigma)
		A = AB[0]
		B = AB[1]
		Z = np.exp(A - B*r)
		return Z


	def hull_white_discount_factors(self, r, dt, theta, kappa, sigma):
		'''
			Returns discount factors following given interest rates path.
			r has simulated instantaneous rates starting from 0 and over dt intervals.
			r should be a numpy array with an interest rate path in each row.
			The second column of the output conatins discount factor for next [0, dt] period.
		'''
		n = r.shape[0]
		print(n)
		m = r.shape[1]
		Z = np.zeros(r.shape)
		Z[:, 0] = 1

		for i in range(1, r.shape[1]):
			t = (i-1)*dt
			T = i*dt
			Z[:, i] = self.hull_white_discount_factor(r[:, i-1], 0, T-t, theta, kappa, sigma)*Z[:, i-1]

		return Z

