#a3_helper_functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm
from scipy.optimize import minimize
from scipy import optimize
from scipy import interpolate

import fixed_income
from utilities import *
import os

class hull_white_fit:
	def __init__(self):
		self.fi = fixed_income.FixedIncome()
		home_dir = os.path.dirname('__file__')
		self.libor_data = self.libor_data_func()
		self.cap_master_df = self.cap_data_func()
		#-----------------------------------------
		# self.opti_params = self.opti_func() #uncomment this section out to run optimization
		# self.opti_kap = self.opti_params[0] 
		# self.opti_vol = self.opti_params[1]
		#---------------------------------------------
		self.opti_kap = 0.11631394289351267 #comment out hard coded values is using optimization
		self.opti_vol = 0.017245499818435216

		self.cap_pricing_df = self.cap_pricing()
		self.theta = self.find_theta()

	def libor_data_func(self):
		home_dir = os.path.dirname(__file__)
        #Preparing LIBOR Data
		self.libor_data = pd.read_csv(home_dir + '/a3_libor.csv',header = 0)
		self.dates = pd.to_datetime(self.libor_data['Date'])
		self.libor_data['Date'] = self.dates
		start_date = self.dates[0]
		self.libor_data['T_30_360'] = np.array(self.dates.apply(lambda x: self.fi.get_days_30I_360(start_date,x)))
		self.libor_data['Discount'] = 1/(1+(self.libor_data['Zero Rate']/100)/2)**(2*self.libor_data['T_30_360'])
		#creap need to get settle date
		self.libor_data['Expiry_day_count'] = np.array(self.dates.apply(lambda x: (x - self.libor_data['Date'][0]).days))
		self.libor_data['T_ACT_360'] = np.array(self.dates.apply(lambda x: self.fi.get_days_act_360(start_date,x)))

		#T_i - T_i-1 where T_i is ACT/360 convention
		self.libor_data['Tau'] = self.libor_data['T_ACT_360'].diff(1)
		forwards = np.array((1/self.libor_data['Tau'])*\
                                ((-self.libor_data['Discount'].diff(1))/self.libor_data['Discount']))

		self.libor_data['Forward'] = forwards

		return self.libor_data
    
	def cap_data_func(self):
		strike = []
		home_dir = os.path.dirname(__file__)
		self.libor_data_func()
		self.cap_master_df = pd.read_csv(home_dir + '/hw_atm_caps.csv',header = 0)
		self.cap_master_df.columns = ['Expiry','Black Imp Vol']
		for m in self.cap_master_df['Expiry']:
			x_n = np.sum(self.libor_data['Tau'][2:4*m +1]*\
				self.libor_data['Forward'][2:4*m +1]*self.libor_data['Discount'][2:4*m +1])/\
				np.sum(self.libor_data['Tau'][2:4*m +1]*self.libor_data['Discount'][2:4*m +1])
			strike.append(x_n)
		self.cap_master_df['ATM Strike'] = strike
		return self.cap_master_df

	def d_1_2(self,flat_vol,forward_libor,t,strike,type = 1):
		d1 = (1/(flat_vol*np.sqrt(t)))*np.log(forward_libor/strike) + 0.5*flat_vol*np.sqrt(t)
		if type == 1:
			return d1
		elif type == 2:
			return d1 - flat_vol*np.sqrt(t)

	def caplet_black(self,time_index,N, flat_vol, strike):
		"""
		time_index - index based on T_30_360 convenion (example - 0.75 -> 3)
		Returns caplet value for a given time based on Black formula.
		"""
		t_i = self.libor_data['Expiry_day_count'][time_index]/365
		tau = self.libor_data['Tau'][time_index +1]
		fwd = self.libor_data['Forward Rate'][time_index+1]/100 #Note used Libor Forward
		fv = flat_vol/100
		discount = self.libor_data['Discount'][time_index+1]
		d1 = self.d_1_2(fv,fwd,t_i,strike)
		d2 = self.d_1_2(fv,fwd,t_i,strike,type =2)
		phi_d1 = norm.cdf(d1)
		phi_d2 = norm.cdf(d2)
		pv = ((N*tau)*discount*(fwd*phi_d1 - strike*phi_d2))
		return pv

	def cap_black(self,maturity_index, N, vol=False):
		"""Returns cap value based on Black formula."""
		cap_master = self.cap_data_func()
		maturity = cap_master['Expiry'][maturity_index]
		if vol == False:
			flat_vol = cap_master['Black Imp Vol'][maturity_index]
		else:
			flat_vol = vol
		strike = cap_master['ATM Strike'][maturity_index]
		caplet_range = np.arange(1,maturity*4)
		cap_pv = 0
		for index in caplet_range:
			cap_pv += self.caplet_black(index,N,flat_vol,strike)
		return cap_pv	

	def caplet_hull_white(self,index,N,vol,kappa, strike_input):
		"""
			Similiar parameters as Black model but needs constant vol and kappa
		"""
		tau = self.libor_data ['Tau'][index+1]
		t_i = self.libor_data['Expiry_day_count'][index]/365
		discount = self.libor_data['Discount'][index+1]
		disc_shift = self.libor_data['Discount'][index]
		b = (1/kappa)*(1 - np.exp(-kappa*tau))
		sigma_p = b*vol*np.sqrt((1- np.exp(-2*kappa*t_i))/(2*kappa))
		d1 = (1/sigma_p)*np.log(discount/(disc_shift*strike_input)) + sigma_p/2
		d2 = d1 - sigma_p
		V = (N/strike_input)*(strike_input*disc_shift*norm.cdf(-d2) - discount*norm.cdf(-d1))
		return V

	def cap_HW(self,maturity_index, vol, N, kappa):
		"""Returns cap value based on HW model."""
		cap_master = self.cap_data_func()
		maturity = cap_master['Expiry'][maturity_index]
		# flat_vol = cap_master['Flat_Vol'][maturity_index]
		strike = cap_master['ATM Strike'][maturity_index]
		caplet_range = np.arange(1,maturity*4)
		cap_pv = 0
		for index in caplet_range:
			tau = self.libor_data['Tau'][index+1]
			strike_input = (1/(1+strike*tau))
			clet = self.caplet_hull_white(index,10000000,vol,kappa,strike_input)
			cap_pv += clet
		return cap_pv
	
	def to_minimize(self,params):
		kappa = params[0]
		fv = params[1]
		res = 0
		for i in range(15):
			err = self.cap_black(i, 10000000) - self.cap_HW(i, fv,10000000, kappa)
			res += err**2
		res = np.sqrt(res)
		self.opti_res = res
		return res

	def opti_func(self):
		print("STARTING OPTIMIZATION----")
		optimum = minimize(self.to_minimize, x0 = [0.20, 0.010])
		print("Optimal Kappa: ", optimum.x[0],"\n", "Optimal Vol: ", optimum.x[1])
		opti_kap = optimum.x[0]
		opti_vol = optimum.x[1]
		self.opti_params = [opti_kap,opti_vol]
		return self.opti_params

	def cap_pricing(self):
		cap_price_list_black = []
		cap_price_list_hull = []
		# opti_vol = self.opti_vol
		for cap_index in range(len(self.cap_master_df)):
			maturity = self.cap_master_df['Expiry'][cap_index]
			flat_vol = self.cap_master_df['Black Imp Vol'][cap_index]
			strike = self.cap_master_df['ATM Strike'][cap_index]
			caplet_range = np.arange(1,maturity*4)
			caplet_black_pv = []
			caplet_hw_pv = []
			for index in caplet_range:
				caplet_black_pv.append(self.caplet_black(index,10000000,flat_vol,strike))
				tau = self.libor_data['Tau'][index+1]
				strike_input = (1/(1+strike*tau))
				caplet_hw_pv.append(self.caplet_hull_white(index,10000000, self.opti_vol, self.opti_kap, strike_input))
			cap_price_list_black.append(round(sum(caplet_black_pv),3))
			cap_price_list_hull.append(round(sum(caplet_hw_pv),3))
		self.cap_master_df['Cap Prices - Black'] = cap_price_list_black
		self.cap_master_df['Cap Prices - Hull'] = cap_price_list_hull
		# if self.show_prints:
  #           print("\n1d) Summary of Cap Pricing", self.cap_master_df, " \n Optimized Kappa: ", opti_kap, "\n", "Optimized Vol: ", opti_vol)

		return self.cap_master_df



        #--------------------------------------------------------------
        # Estimate theta(t) from today t_0 to 30-years with timestep 1/12
        #--------------------------------------------------------------

 
	def find_theta(self):
		disc_quart = np.array(self.libor_data['Discount']).astype(float)
		x_quart = np.array(self.libor_data['T_ACT_360'])
		x_monthly = np.arange(0,30 + (1/12),(1/12))
		disc_monthly_f = interpolate.interp1d(x_quart, disc_quart, kind = 'cubic')
		disc_monthly = disc_monthly_f(x_monthly)

       #Calculating Monthly Forwards and derivative using finite differencing
		fm = np.array([-(np.log(disc_monthly[i]) - np.log(disc_monthly[i-1]))/(1/12) for i in range(1,len(disc_monthly))]).astype(float)
		d_fm = np.array([(fm[i] - fm[i-1])/(1/12) for i in range(1,len(fm))]).astype(float)

		kappa = self.opti_kap
		sigma = self.opti_vol

		#Calculating Theta
		self.theta = d_fm + kappa*fm[1:] + ((sigma**2)/(2*kappa))*(1 - np.exp(-2*kappa*(x_monthly[2:])))
		return self.theta

	def simulate_interest_rates(self, n):
        # The subscript _A denotes that is a tuple containing antithetic paths in each of the two positions
		dt = 1/12
		r0 = self.fi.hull_white_instantaneous_spot_rate(0, 3*dt, self.libor_data.loc[1, 'Discount'], self.theta, self.opti_kap, self.opti_vol)
		simulated_rates_A = self.fi.hull_white_simulate_rates_antithetic(n, r0, dt, self.theta, self.opti_kap, self.opti_vol)
		#simulated_Z_A = self.fi.hull_white_discount_factors_antithetic_path(simulated_rates_A, dt)
		return simulated_rates_A

	def calculate_T_year_rate_APR(self, r_A, lag, horizon, previous_rates):
		'''
            Intended to return the 10-year Treasury rate at the end of every pool with a lag of 3 months (horizon is 10).
            Returns a list where in each position (corresponding to each ending month in end) has a tupple with the (antithetic) APR
            r_A contains antithetic simulations of the instantaneous spot rate.
            lag is the lag in months.
            horizon is the horizon in years of the period to get the APR.
            previous rates are the T years Treasury rates in BEY 
        '''
		previous_rates = ((1+(np.array(previous_rates)/2))**(2/12)-1)*12 # From BEY to monthly APR
		n = r_A.shape[0]
		Z_A = self.fi.hull_white_discount_factor(r_A, 0, horizon, self.theta, self.opti_kap, self.opti_vol)
		r_APR = 12*((1/Z_A)**(1/(12*horizon)) - 1)
		r_APR[:, lag:] = r_APR[:, :-lag]
		r_APR[:, :lag] = np.array(previous_rates[-lag:])
		return r_APR




if __name__ == '__main__':
    test = hull_white_fit()
    print(test.cap_master_df)
    print(test.theta)
    print(test.simulate_interest_rates(10))
    # print(test.opti_params)
   # t2 = test.libor_data_func()
   # t3 = test.cap_data_func()
    # print(t2.tail())
   # print(t3.tail())