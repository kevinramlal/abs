
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve
from datetime import datetime
import time
import calendar
import os

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy.stats as stats

import sys
import fixed_income
from utilities import *


import warnings
warnings.filterwarnings("ignore")

class BART:
	'''
		Inputs:
		 tranche_list: List of Tranche names
		 tranche_principal: List of Tranche initial principal
		 bond_spread: list of spreads per bond: Rates (simulated) + spread 
		 ridership_forecast:  Ridership Forecasts as np.array #to be added into class [N,T]
		 rev_forecasts: Revenue Forecasts as np.array #to be added into class [N,T] N number of sumulations, T is number of months 
		 rev_percentage: % of Revenue that goes toward paying tranches
		 simulated_rates: Simulated Rates Monte Carlo [N,T]
		 maturity: Number of months until maturity  #all tranches have same maturity 

		 TODO: Is LTV needed? I assume it's 100%

	'''

	def __init__(self, tranche_list, tranche_principal, bond_spread, base_coupon_rate, rev_percentage, simulated_rates, maturity, tables_file, show_prints=False, show_plots=False):
		self.tranche_list = tranche_list
		self.tranche_principal = tranche_principal
		self.bond_spread = bond_spread
		self.rev_percentage = rev_percentage
		self.base_coupon_rate = base_coupon_rate
		self.simulated_rates = simulated_rates
		self.T = maturity #the input should be already in the form of Months 
		self.show_prints = show_prints
		self.show_plots = show_plots
		self.tables_file = tables_file
		self.fi = fixed_income.FixedIncome()
	
		#Processed Values 
		self.N = simulated_rates.shape[0] # Number of simulations
		self.regular_classes = self.tranche_list[:]
		if 'R' in self.regular_classes: #might be kept 
			self.regular_classes.remove('R')

	def forecast_ridership(self):

		# ------------------------------ #
		# Data collection
		# --------------------------------#

		home_dir = os.path.dirname(__file__)
		bart_fcst = pd.read_csv(home_dir +'/BART_ridership_forecast.csv')
		riders = {}
		year_min = 2015
		year_max = 2018
		periods = []

		print("Collecting data...\n")
		for year in range(year_min, year_max+1):
		    folder = 'ridership_'+str(year)
		    for month in range(12):
		        month_str = ('0'+str(month+1))[-2:]
		        sheet1 = 'Weekday OD'
		        sheet2 = 'Saturday OD'
		        sheet3 = 'Sunday OD'
		        if year == 2018:
		            filename = 'Ridership_'+str(year)+month_str
		            if month+1 >= 2:
		                sheet1 = 'Avg Weekday OD'
		                sheet2 = 'Avg Saturday OD'
		                sheet3 = 'Avg Sunday OD'                
		        else:
		            filename = 'Ridership_'+calendar.month_name[month+1]+str(year)
		        path_to_file = home_dir +'/'+folder+'/'+filename+'.xlsx'
		        #print(path_to_file)
		        xlsx = pd.ExcelFile(path_to_file)
		        df1 = pd.read_excel(xlsx, sheet1, skiprows=1)
		        df2 = pd.read_excel(xlsx, sheet2, skiprows=1)
		        df3 = pd.read_excel(xlsx, sheet3, skiprows=1)
		        key = str(year) + month_str
		        periods.append(key)
		        riders[key] = [df1, df2, df3]

		# ------------------------------ #
		# Data processing
		# --------------------------------#

		last_period = str(year_max)+'12'
		columns = riders[last_period][0].columns[1:-1]

		# Weekday entries
		wk_ent_np = np.zeros(((year_max-year_min+1)*12, len(columns)))
		# Saturday entries
		st_ent_np = np.zeros(((year_max-year_min+1)*12, len(columns)))
		# Sunday entries
		sn_ent_np = np.zeros(((year_max-year_min+1)*12, len(columns)))

		for t, period in enumerate(periods):
		    for c, col in enumerate(columns):
		        df1 = riders[period][0]
		        if col in df1:
		            wk_ent_np[t, c] =  riders[period][0][col].iloc[-1]
		            st_ent_np[t, c] =  riders[period][1][col].iloc[-1]
		            sn_ent_np[t, c] =  riders[period][2][col].iloc[-1]

		wk_ent = pd.DataFrame(wk_ent_np, columns=columns)
		st_ent = pd.DataFrame(st_ent_np, columns=columns)
		sn_ent = pd.DataFrame(sn_ent_np, columns=columns)

		wk_ent['period'] = periods
		st_ent['period'] = periods
		sn_ent['period'] = periods

		wk_ent = wk_ent.set_index('period')
		st_ent = st_ent.set_index('period')
		sn_ent = sn_ent.set_index('period')

		# Total week entries
		tt_ent = 5*wk_ent + st_ent + sn_ent
		total = pd.DataFrame(tt_ent.iloc[:,:-5].sum(axis=1), columns=['old'])
		total['new'] = tt_ent.iloc[:,-5:].sum(axis=1)

		# ------------------------------ #
		# Forecast
		# --------------------------------#

		old_diff = total['old'].diff()[1:]
		model = SARIMAX(old_diff, order=(2,0,0), seasonal_order=(1,1,0,12))
		model_fit = model.fit(disp=0)
		resid = model_fit.resid
		total_last = total['old'].iloc[-1] + total['new'].iloc[-1]
		pred_diff = np.array(model_fit.forecast(self.T))
		pred_total_old = np.array(total['old'].iloc[-1] + np.cumsum(pred_diff))
		pred_diff_adj = np.zeros(len(pred_diff))

		prev = 0
		pred_diff_cum = np.cumsum(pred_diff)
		for i in range(int(self.T/12)):
		    bart_pchange = 0
		    bart_weight = 1
		    last_riders = total_last + prev
		    if i in bart_fcst.index:
		        bart_pchange = bart_fcst.loc[i, 'Annual Change']
		        bart_weight = bart_fcst.loc[i, 'Weight']
		    model_change = pred_diff_cum[12*(i+1)-1] - prev
		    bart_change = last_riders*bart_pchange
		    prev = pred_diff_cum[12*(i+1)-1]
		    final_change = bart_change*bart_weight + model_change*(1-bart_weight)
		    pred_diff_adj[12*i:12*(i+1)] = pred_diff[12*i:12*(i+1)] + (final_change - model_change)/12

		history = len(total['old'])
		hist_total = (total['old']+ total['new'])*30/7
		pred_total_adj = (total_last + np.cumsum(pred_diff_adj))*30/7

		# ------------------------------ #
		# Simulations
		# ------------------------------ #

		# Normal parameters
		mu, sigma = stats.norm.fit(np.array(resid))

		n_sim = 1000
		innovations = stats.norm.rvs(loc=0, scale=sigma, size=(n_sim, self.T))
		sim_diff = innovations + pred_diff_adj
		self.ridership_forecast = (total_last + np.cumsum(sim_diff, axis=1))*30/7 #we get the forecast here then 

		# ------------------------------ #
		# Graphs
		# --------------------------------#

		if self.show_plots:
			fig, ax = plt.subplots(2,2, figsize=(15,10))
			fontsize = 16
			fontsize_leg = 12
			data_plot = np.array(resid)/1e3

			# Time series
			ax[0,0].plot(data_plot)
			ax[0,0].set_title('Time series of residual', fontsize=fontsize)
			ax[0,0].set_xlabel('Month', fontsize=fontsize)
			ax[0,0].set_ylabel('Riders (thousands)', fontsize=fontsize)

			# Histogram
			ax[0,1].hist(data_plot, bins=20)
			ax[0,1].set_title('Histogram of residual', fontsize=fontsize)
			ax[0,1].set_xlabel('Riders (thousands)', fontsize=fontsize)
			ax[0,1].set_ylabel('Frequency', fontsize=fontsize)

			# ACF
			plot_acf(data_plot, ax=ax[1,0])
			ax[1,0].set_title('ACF of residual', fontsize=fontsize)
			ax[1,0].set_xlabel("Lag", fontsize=fontsize)
			ax[1,0].set_ylabel("Autocorrelation", fontsize=fontsize)

			# QQplot
			sm.qqplot(data_plot, stats.norm, loc=mu, scale=sigma, line='45', fit=True, ax=ax[1,1])
			ax[1,1].set_title('Normal', fontsize=fontsize)
			ax[1,1].set_xlabel("Theoretical Quantiles", fontsize=fontsize)
			ax[1,1].set_ylabel("Sample Quantiles", fontsize=fontsize)
			fig.tight_layout()
			fig.savefig('residual.png')

			#--------------------------

			# Expected forecast
			fig, ax = plt.subplots(figsize=(15,7))		
			ax.plot(np.arange(history), hist_total, label='History')
			ax.plot(np.arange(history, history+self.T), pred_total_adj, label='Forecast')
			ax.set_title('Total Ridership', fontsize=fontsize)
			ax.set_xlabel('Month', fontsize=fontsize)
			ax.set_ylabel('Riders', fontsize=fontsize)
			ax.legend(loc = 'upper left', fontsize = fontsize_leg)
			fig.tight_layout()
			fig.savefig('expected_forecast.png')

			#--------------------------

			# Sample simulations
			fig, ax = plt.subplots(figsize=(15,7))
			ax.plot(np.arange(history), hist_total, label='History')
			ax.plot(np.arange(history, history+self.T), pred_total_adj, label='Expected Forecast')
			ax.plot(np.arange(history, history+self.T), self.ridership_forecast[0], label='Simulation 1')
			ax.plot(np.arange(history, history+self.T), self.ridership_forecast[2], label='Simulation 2')
			ax.set_title('Total Ridership', fontsize=fontsize)
			ax.set_xlabel('Month', fontsize=fontsize)
			ax.set_ylabel('Riders', fontsize=fontsize)
			ax.legend(loc = 'upper left', fontsize = fontsize_leg)
			fig.tight_layout()
			fig.savefig('sample_simulations.png')


			plt.show()

	def forecast_revenue(self):
		# Forecast Ridership
		self.forecast_ridership()

		# Fairs deterministic
		#last_fair = 481.8/120.554 # FY18
			# FY18 increase 2.7%
			# FY16 increase 3.4%
		#p_increase_2y = 0.02
		#fares = np.ones(self.T)
		#fares[1::2] = 1 + p_increase_2y
		#fares = last_fair*np.cumprod(fares)
		#self.revenue = self.ridership_forecast*fares.reshape(1,-1)

		# Fairs stochastic
		last_fair = 481.8/120.554 # FY18
			# FY18 increase 2.7%
			# FY16 increase 3.4%
		fares_high = np.array([4.0787434, 3.2293254, 3.4738840, 3.6512020, 3.5051721])
		fares_low =  np.array([0.4147587, 0.4195397, 0.6085524, 1.5142361, 0.7104166, 0.0599787])
		np.random.seed(0)
		Th = int(self.T/12/2)
		rand_high = np.random.choice(fares_high, size=(self.N, Th))
		rand_low = np.random.choice(fares_low, size=(self.N, Th+1))
		fares_year = np.ones((self.N, Th*2+1))
		fares_year[:, 1::2] = 1 + rand_high/100
		fares_year[:, 0::2] = 1 + rand_low/100
		fares_year = last_fair*np.cumprod(fares_year, axis=1)
		fares = np.repeat(fares_year, 12, axis=1)
		self.revenue = self.ridership_forecast*fares

	def calculate_cashflows(self):
		"""
			Given tranche information, and revenue, calculate cashflows per trance
		"""
		self.bond_spread_dict ={self.regular_classes[i]:self.bond_spread[i] for i in range(len(self.regular_classes))}
		self.residual = np.zeros((self.N,self.T))

		#Tranches
		self.bonds_balance = {k:np.zeros((self.N,self.T)) for k in self.regular_classes} #initialize our bonds principal
		self.bonds_interest = {k:np.zeros((self.N,self.T)) for k in self.regular_classes} #keep track of interest
		for i in range(len(self.regular_classes)):
			self.bonds_balance[self.regular_classes[i]][:,0] = self.tranche_principal[i] #This intializes all simulations 

		#Cashflow Tracking
		self.bonds_interest_cf = {k:np.zeros((self.N,self.T)) for k in self.regular_classes}
		self.bonds_amort_cf = {k:np.zeros((self.N,self.T)) for k in self.regular_classes}
		self.bonds_prepay_cf = {k:np.zeros((self.N,self.T)) for k in self.regular_classes}

		#OK! Time for cash flows
		for month in range(1,self.T):
			pmt = self.revenue[:,month]*self.rev_percentage #monthly revenue x % for ALL simulations

			#FIRST PASS
			for cl in self.regular_classes:
				prev_balance = self.bonds_balance[cl][:,month-1] #all simulations array for month
				r_month = (self.base_coupon_rate + self.bond_spread_dict[cl])/12 # self.simulated_rates[:,month-1]
				amortized_pmt = self.coupon_payment(r_month, self.T - month , prev_balance) #This should work as numpy array component wise multiplication
				interest_accrued = prev_balance*r_month #also numpy array multiplication 
				
				#Water Fall  
				pmt, self.bonds_balance[cl][:,month], self.bonds_interest[cl][:,month], \
				 self.bonds_interest_cf[cl][:,month], self.bonds_amort_cf[cl][:,month]  = self.first_pass(pmt,prev_balance,amortized_pmt,interest_accrued)

			#SECOND PASS
			#Principal prepayment  
			for cl in self.regular_classes:
				balance_after_waterfall = self.bonds_balance[cl][:,month]
				pmt, self.bonds_balance[cl][:,month], self.bonds_prepay_cf[cl][:,month] = self.second_pass(pmt,balance_after_waterfall)

			
			self.residual[:,month] = pmt #should be all zeroes unless theres left over cash after paying off ALL tranches 

		return self.bonds_balance, self.bonds_interest, self.bonds_interest_cf, self.bonds_amort_cf, \
			self.bonds_prepay_cf, self.residual

	def first_pass(self,pmt_array,prev_balance_array,amort_pmt_array,interest_accrued_array):
		"""
		pmt_array should be array of length N for a given month
		prev_balance array: array of length N for given month
		amort_pmt_array: array of length N
		interest_accrued_array: array of Length n

		Vectorized 
		"""
		# interest_accrued = []
		# new_balance = []
		# pmt_new = []

		# np.max(interest_accrued_array - pmt_array,0)
		water_fall_payer = lambda x: max(0,x) #x should be Interest - Payment for interest, then Amort Bal - Payment for principal
		pmt_after_pay = lambda x: max(0,x) #reverse order of operations as previous 

		# interest_accrued = np.array(list(map(water_fall_payer,(interest_accrued_array - pmt_array))))
		interest_accrued = np.maximum((interest_accrued_array - pmt_array),0)
		pmt_after_interest = np.maximum((pmt_array - interest_accrued_array),0)
		interest_cf = np.minimum(pmt_array,interest_accrued_array) 
		# pmt_after_interest = np.array(list(map(pmt_after_pay,(pmt_array - interest_accrued_array))))

		# amort_pay_deduct =  np.array(list(map(water_fall_payer,(amort_pmt_array - pmt_after_interest)))) #Either 0 
		amort_pay_deduct = np.maximum((amort_pmt_array - pmt_after_interest),0)
		pmt_after_princpal = np.maximum((pmt_after_interest - amort_pmt_array),0)
		amort_cf = np.minimum(pmt_after_interest,amort_pmt_array)

		# pmt_after_princpal = np.array(list(map(pmt_after_pay, (pmt_after_interest - amort_pmt_array))))
		new_balance  = prev_balance_array - amort_pmt_array + amort_pay_deduct + interest_accrued #(Prev Balance - Amortization + Deduction from Amort (either 0 or amort - pmt)) + Interest Accrus 

		#Case 1: PMT > AMORTIZATION
		#amort_pay_deduct = 0
		#new balance = old balance - amortization amount

		#case 2: PMT < AMORT
		#amort_pay_deduct = amort_pmt_array - pmt_after_interest (example 1000 - 400 = 600)
		#NEWB = old - amort amount + (amort aount - pmt) = old - pmt 

		#UNVECTORIZED CODE TO KEEP TRACK OF WHATS GOING ON
		# for i in range(len(pmt_array)): #loop through interatation 
		# 	int_acc = interest_accrued_array[i]
		# 	prev_bal = prev_balance_array[i]
		# 	pmt_sim = pmt[i]
		# 	amort_pmt = amort_pmt_array[i]

		# 	#first interest payments
		# 	if pmt_sim - int_acc > 0:
		# 		pmt_sim -= int_acc
		# 		int_acc = 0
		# 	else:
		# 		int_acc -= pmt_sim
		# 		pmt_sim = 0

		# 	if pmt_sim - amort_pmt >0:
		# 		pmt_sim -= amort_pmt
		# 		new_bal = prev_bal - amort_pmt
		# 	else:
		# 		new_bal = prev_bal - pmt_sim
		# 		pmt_sim = 0

		# 	interest_accrued.append(int_acc)
		# 	pmt_new.append(pmt_sim)
		# 	new_balance.append(new_bal + int_acc)

		pmt_new = pmt_after_princpal
		return pmt_new,new_balance,interest_accrued, interest_cf, amort_cf

	def second_pass(self,pmt_array,balance_array):
		"""
		TODO: Vectorize 
		"""
		# pmt_new = []
		# for j in range(len(pmt_array)):
		# 	pmt_sim = pmt_array[j]

		#UNVECTORIZED 
		# 	i = 0
		# 	while pmt_sim > 0:
		# 		pmt_temp = max(0,pmt_sim - self.bonds_balance[self.regular_classes[i]][j,month]) #Remaining pmt is either 0 or left over from tranches in priority order
		# 		self.bonds_balance[self.regular_classes[i]][j,month] = max(0,self.bonds_balance[self.regular_classes[i]][j,month] - pmt_sim) #Update balance
		# 		pmt_sim = pmt_temp
		# 		i += 1
		# 	pmt_new.append(pmt_sim) #should all be zeroes theoretically unless extra left over afte ALL tranche principals paid off 

		new_cf = np.minimum(pmt_array,balance_array) #either full balance or full pmt 
		new_balance = np.maximum((balance_array - pmt_array),0) #if enough money to pay off balance, then 0, else difference
		new_pmt = np.maximum((pmt_array - balance_array),0) #new pmt is residual after removing balance, or 0
		return new_pmt, new_balance, new_cf 
	
	def coupon_payment(self, r_month, months_remaining, balance):
		return r_month*balance/(1-1/(1+r_month)**months_remaining)


	def calculate_bond_prices(self, dr=0, show_prints=False):

		r = self.simulated_rates + dr
		
		Nh = (int)(self.N/2)

		# Calculatesimulated prices
		bonds_simulated_prices = {}
		results = np.zeros((len(self.regular_classes), 3))
		for i in range(len(self.regular_classes)):
			cl = self.regular_classes[i]
			r_class = r + self.bond_spread_dict[cl]
			Z = self.fi.hull_white_discount_factors_antithetic_path(r_class, dt=1/12)[:, :self.T]
			bonds_prices = np.sum((self.bonds_interest_cf[cl] + self.bonds_amort_cf[cl] + self.bonds_prepay_cf[cl])*Z, axis=1)
			bonds_simulated_prices[cl] = (bonds_prices[:Nh] + bonds_prices[Nh:])/2

			# Price
			results[i, 0] = np.mean(bonds_simulated_prices[cl])
			results[i, 1] = np.std(bonds_simulated_prices[cl])
			results[i, 2] = results[i, 1]/np.sqrt(Nh)

		results_df = pd.DataFrame(results, columns=['Average Price', 'Std. Deviation', 'Std. Error'])
		results_df.index = self.regular_classes

		if show_prints:
			self.tables_file.write(latex_table(results_df, caption = "Simulated Prices", label = "prices", index = True))
			print("Results :\n\n ",results_df)
			print("Residual : {:,}".format(1566000000 - results_df['Average Price'].sum()))

		return [abs(float(1566000000 - results_df['Average Price'].sum())), results_df]

	def calculate_duration_convexity(self):

		dr = 0.0001
		simulation_summary = self.calculate_bond_prices(dr=0)[1]
		simulation_summary_up = self.calculate_bond_prices(dr=dr)[1]
		simulation_summary_dn = self.calculate_bond_prices(dr=-dr)[1]

		dur = np.zeros(len(self.regular_classes))
		conv = np.zeros(len(self.regular_classes))

		P = simulation_summary['Average Price']
		P_up = simulation_summary_up['Average Price']
		P_dn = simulation_summary_dn['Average Price']

		dur = (P_dn-P_up)/(P*2*dr)
		conv = (P_dn+P_up-2*P)/(P*dr**2)

		dur_conv = np.zeros((2,len(self.regular_classes)))
		dur_conv[0, :] = dur
		dur_conv[1, :] = conv

		dur_conv = pd.DataFrame(dur_conv.T, columns = ['Duration', 'Convexity'])
		dur_conv.index = self.regular_classes

		self.tables_file.write(latex_table(dur_conv.round(2), caption = "Duration and Convexity", label = "dur_conv", index = True))
		print('\n')
		print(dur_conv)


#TODO: Duration, Convexity, OAS - to make it par
#TODO: CDS Pricing
#TODO: 





