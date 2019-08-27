
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

	def hull_white_A_B(self, t, T, kappa, theta, sigma):
		'''
			Calculates A(t,T) in Hull and White
		'''

		B = self.hull_white_B(t, T, kappa)
		def hull_white_integrand(tau, dt, kappa, theta):
			return self.hull_white_B(tau, T, kappa)*self.hull_white_theta(tau, dt, theta)

		dt = 1/12
		I = quad(hull_white_integrand, t, T, args=(dt, kappa, theta))[0]
		A = -I + (sigma**2 / (2.0*kappa**2))*(T - t + (1.0 - np.exp(-2.0*kappa*(T - t)))/(2.0*kappa) - 2.0*B)
		return (A, B)

	def hull_white_instantaneous_spot_rate(self, t, T, Z, theta, kappa, sigma):
		'''
			Given a discount factor Z for the [t, T] period, it returns the implicit instantaneous spot rate
		'''
		AB =  self.hull_white_A_B(t, T, kappa, theta, sigma)
		A = AB[0]
		B = AB[1]
		r0 = -(np.log(Z)-A)/B
		return r0

	def hull_white_simulate_rates(self, n, r0, theta, kappa, sigma):
		'''
			Simulates n paths of instantaneous rates using the Hull and White model
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

		print(r)

