
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve
from datetime import datetime
import time
import sys
from utilities import *

import warnings
warnings.filterwarnings("ignore")

class REMIC:
	'''
		Values REMIC bonds and calculates relevant metrics.
	'''

	def __init__(self, start_date, first_payment_date, pool_interest_rate, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay, simulated_rates, simulated_Z, show_prints=False, show_plots=False):
		# Direct inputs
		self.start_date = datetime.strptime(start_date, "%m/%d/%Y")
		self.first_payment_date = datetime.strptime(first_payment_date, "%m/%d/%Y")
		self.pool_interest_rate = pool_interest_rate
		self.pools_info = pools_info
		self.classes_info = classes_info
		self.principal_sequential_pay = principal_sequential_pay
		self.accruals_sequential_pay = accruals_sequential_pay
		self.simulated_rates = simulated_rates
		self.simulated_Z = simulated_Z
		self.show_prints = show_prints
		self.show_plots = show_plots

		# Processed attributes
		self.maturity = np.max(self.pools_info['Term'])
		self.n_pools = self.pools_info.shape[0]
		self.classes = list(self.classes_info['REMIC Classes'])
		self.regular_classes = self.classes[:]
		self.regular_classes.remove('R')
		self.accrual_classes = list(self.accruals_sequential_pay.keys())
		self.principal_classes = list(set(self.regular_classes) - set(self.accrual_classes))

		self.pools_info['Original Balance'] = self.pools_info['Original Balance'].astype(float)
		self.pools_info['WAC'] = self.pools_info['WAC'].astype(float)
		self.classes_info['Original Balance'] = self.classes_info['Original Balance'].astype(float)
		self.classes_info['Class Coupon'] = self.classes_info['Original Balance'].astype(float)
		self.classes_info.index = self.classes_info['REMIC Classes']

		self.calculate_pool_groups_proportions()


	def simulation_result(self, hazard_model, simulated_lagged_10_year_rates_A,part_price,part_convexity,data_type):

		#---------------------------
		# Cashflows and Prices
		#---------------------------

		SMM = self.calculate_pool_simulation_prepayment(hazard_model, simulated_lagged_10_year_rates_A)
		N = simulated_lagged_10_year_rates_A.shape[0]
		Nh = int(N/2)
		prices_all = np.zeros((N, len(self.classes)))
		prices_paired = np.zeros((Nh, len(self.classes)))
		avg_cf = np.zeros((self.maturity,len(self.classes)))
		oas_class = 'CA'
		oas_class_cf = np.zeros((N, self.maturity))
		for n in range(N):
			print("Simulation: " + str(n+1) + "/" + str(N))
			r_n = self.simulated_rates[n]
			Z_n = self.simulated_Z[n]
			SMM_n = [SMM[i][n] for i in range(self.n_pools)]
			pool_summary_n = self.calculate_pool_cf(SMM_n)
			total_cf = self.calculate_classes_cf(pool_summary_n, r_n)
			oas_class_cf[n] = total_cf[oas_class]
			avg_cf = avg_cf + total_cf
			prices_all[n] = self.price_classes(total_cf, Z_n)
		avg_cf = avg_cf/N

		# Pair antithetic prices
		for n in range(Nh):
			prices_paired[n] = (prices_all[n] + prices_all[n+Nh])/2

		summary_np = np.zeros((3,len(self.classes)))
		summary_np[0, :] = prices_paired.mean(axis=0)
		summary_np[1, :] = prices_paired.std(axis=0)
		summary_np[2, :] = summary_np[1, :]/np.sqrt(Nh)

		self.simulation_summary = pd.DataFrame(summary_np.T, columns = ['Average price', 'Standard Deviation', 'Standard error'])
		self.simulation_summary.index = self.classes

		if self.show_prints:
			print('\nPart '+ part_price + ':\n' + str(self.simulation_summary) + '\n')
			#print(latex_table(self.simulation_summary, caption = "Simulation summary", label = "prices", index = True))

		#---------------------------
		# Duration and Convexity
		#---------------------------

		dur_conv = self.calculate_durations_and_convexities(avg_cf, dr=0.0001, dt=1/12)

		if self.show_prints:
			print('\nPart '+ part_convexity + ':\n' + str(dur_conv) + '\n')
			#print(latex_table(dur_conv, caption = "Duration and Convexity", label = "duration", index = True))

		#---------------------------
		# OAS
		#---------------------------

		dt = 1/12
		par_value = self.classes_info.loc[oas_class, 'Original Balance']
		# Spot rates are continously compounded.
		# Accoding to the lectures, OAS is the spread using monthly compounding.
		monthly_compounded_rates = np.exp(self.simulated_rates*dt)-1
		T = self.maturity
		oas = self.calculate_OAS(par_value, oas_class_cf[:Nh,:T], monthly_compounded_rates[:Nh,:T])

		if self.show_prints:
			print("\nPart G w/ " + data_type + ":\nOAS for " + str(oas_class) + " = " + str(oas*100) + '%\n')


		#---------------------------
		# Hazard Rate
		#---------------------------

		avg_hz = np.mean(SMM[0], axis=0)

		if self.show_plots:
			plt.plot(avg_hz)
			plt.xlabel("Months")
			plt.ylabel("Average Hazard Rate")
			plt.show()



	def calculate_pool_simulation_prepayment(self, hazard_model, simulated_lagged_10_year_rates_A):
		'''
			Receives a fitted hazard_model from the Hazard class.
			Returns numpy array with SMM where rows indicate simulation path and columns indicate month.
		'''
		# Summer variable
		month_start = self.start_date.month
		t = np.arange(0,self.maturity+1)
		T = len(t)
		N = simulated_lagged_10_year_rates_A.shape[0]
		month_index = np.mod(t + month_start - 1, 12) + 1
		summer = np.array([1 if i>=5 and i<=8 else 0 for i in month_index])

		# Coupon gap
		cpn_gap = []
		# cpn_gap is a list of numpy arrays containing the coupon gap for every simulation and month.
		# The list has one numpy array for each pool.
		# Every row indicates a simulation and every column a month.
		for pool_index in range(self.n_pools):
			cpn_gap.append(self.pools_info.loc[pool_index, 'WAC'] - simulated_lagged_10_year_rates_A*100)

		# Prepayment
		SMM = [np.zeros((N, T))]*self.n_pools
		for pool_index in range(self.n_pools):
			for n in range(N):
				cpn_gap_n = cpn_gap[pool_index][n][:T]
				covars = np.array((cpn_gap_n, summer)).T
				SMM[pool_index][n] = hazard_model.calculate_prepayment(t, covars)

		return SMM

	def calculate_pool_cf(self, SMM):
		'''
			If harzard_flag = True, then the calculated hazard array must be inputted
			When flag is true, this will replace the SMM method
		'''
		columns = ['Total Principal', 'Total Interest', 'Balance', 'Interest Available to CMO']
		pool_summary = pd.DataFrame(np.zeros((self.maturity+1, 4)), columns = columns)
		pools = []

		for pool_index in range(self.n_pools):
			balance = self.pools_info.loc[pool_index, 'Original Balance']
			r_month = self.pools_info.loc[pool_index, 'WAC']/12/100
			term = self.pools_info.loc[pool_index, 'Term']
			age = self.pools_info.loc[pool_index, 'Age']
			columns = ['PMT', 'Interest', 'Principal', 'CPR', 'SMM', 'Prepay_CF', 'Balance']
			pool = pd.DataFrame(np.zeros((self.maturity+1,7)), columns = columns)
			pool.loc[0,'Balance'] = balance
			pool['SMM'] = SMM[pool_index]
			for month in range(1, term+1):
				prev_balance = pool.loc[month-1,'Balance']
				pool.loc[month, 'PMT'] = self.coupon_payment(r_month, term - (month - 1), prev_balance)
				pool.loc[month, 'Interest'] = prev_balance*r_month
				pool.loc[month, 'Principal'] = prev_balance if pool.loc[month, 'PMT'] - pool.loc[month, 'Interest'] > prev_balance else pool.loc[month, 'PMT'] - pool.loc[month, 'Interest']
				pool.loc[month, 'Prepay_CF'] = pool.loc[month, 'SMM']*(prev_balance - pool.loc[month, 'Principal'])
				pool.loc[month, 'Balance'] = prev_balance - pool.loc[month, 'Principal'] - pool.loc[month, 'Prepay_CF']
			pools.append(pool)

		for pool in pools:
			pool_summary['Total Principal'] += pool['Principal'] + pool['Prepay_CF']
			pool_summary['Total Interest'] += pool['Interest']
			pool_summary['Balance'] += pool['Balance']

		for month in range(1, self.maturity+1):
			pool_summary.loc[month, 'Interest Available to CMO'] = self.pool_interest_rate/12*pool_summary.loc[month - 1, 'Balance']

		return pool_summary

	def coupon_payment(self, r_month, months_remaining, balance):
		return r_month*balance/(1-1/(1+r_month)**months_remaining)

	def calculate_pool_groups_proportions(self):
		self.principal_groups_proportions = {}
		total_balance = 0
		for group in self.principal_sequential_pay:
			for cl in self.principal_sequential_pay[group]:
				self.principal_groups_proportions[group] = self.principal_groups_proportions.get(group, 0) + self.classes_info.loc[cl, 'Original Balance']
				total_balance += self.classes_info.loc[cl, 'Original Balance']
		for group in self.principal_groups_proportions:
			self.principal_groups_proportions[group] = float(self.principal_groups_proportions[group])/total_balance

	def calculate_classes_cf(self, pool_summary, r):
		'''
			Calculates cash flows of all bonds given the simulated interest rates.
			First cash flow starting at month 1.
		'''
		columns = self.classes
		classes_balance = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)
		classes_interest = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)
		classes_accrued = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)
		classes_principal = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)
		classes_interest_cf = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)

		# Initial Balance
		for cl in self.classes:
			classes_balance.loc[0, cl] = self.classes_info.loc[cl, 'Original Balance']

		for month in range(1, pool_summary.shape[0]):

			# Interest
			for cl in self.accrual_classes:
				classes_interest.loc[month, cl] = self.pool_interest_rate/12*classes_balance.loc[month - 1, cl]

			# Distribute Principal
			principal = pool_summary.loc[month, 'Total Principal']
			for group in self.principal_groups_proportions:
				principal_group_remaining = principal*self.principal_groups_proportions[group]
				for cl in self.principal_sequential_pay[group]:
					payment = min(principal_group_remaining, classes_balance.loc[month - 1, cl])
					classes_principal.loc[month, cl] += payment
					principal_group_remaining -= payment

			# Distribute Interest
			for cl in self.accrual_classes:
				interest_remaining = classes_interest.loc[month, cl]
				for cl_prin in self.accruals_sequential_pay[cl]:
					payment = min(interest_remaining, classes_balance.loc[month - 1, cl_prin] - classes_principal.loc[month, cl_prin])
					classes_principal.loc[month, cl_prin] += payment
					interest_remaining -= payment

				last_class = self.accruals_sequential_pay[cl][-1]
				if classes_balance.loc[month - 1, last_class] - classes_principal.loc[month - 1, last_class] > 0:
					classes_principal.loc[month, cl] += interest_remaining
					classes_accrued.loc[month, cl] = classes_interest.loc[month, cl]
				else:
					classes_principal.loc[month, cl] += min(interest_remaining, classes_principal.loc[month, last_class])
					classes_accrued.loc[month, cl] = min(classes_interest.loc[month, cl], classes_principal.loc[month, last_class])

			# Update Balance
			for cl in self.regular_classes:
				classes_balance.loc[month, cl] = max(0, classes_balance.loc[month - 1, cl] + classes_accrued.loc[month, cl] - classes_principal.loc[month, cl])

			# Interest cash flow
			for cl in self.regular_classes:
				if cl in self.accrual_classes:
					classes_interest_cf.loc[month, cl] = classes_interest.loc[month, cl] - classes_accrued.loc[month, cl]
				else:
					classes_interest_cf.loc[month, cl] = self.pool_interest_rate/12*classes_balance.loc[month - 1, cl]

		# Total cash flow
		total_interest = classes_interest_cf.sum(1)
		total_cf = classes_principal + classes_interest_cf
		coupon_differential = pool_summary['Total Principal'] + pool_summary['Interest Available to CMO'] - total_cf.iloc[:,0:-1].sum(axis=1)
		total_cf['R'] = coupon_differential + total_cf.iloc[:,0:-1].sum(axis=1)*r[:total_cf.shape[0]]/12/2 # Quick approximation

		return total_cf.iloc[1:,:]

	def price_classes(self, total_cf, Z):

		prices = np.zeros(len(self.classes))
		m = total_cf.shape[0]
		for cl_ind in range(len(self.classes)):
			cashflows = np.array(total_cf.iloc[:, cl_ind])
			prices[cl_ind] = np.sum(cashflows*Z[:m])
		return prices

	def calculate_price_given_yield(self, y, total_cf, cl, dt):
		'''
			Continuous compounding.
			cl is the class
		'''
		m = total_cf.shape[0]
		Z = np.exp(-y*np.arange(1, m+1)*dt)
		cashflows = np.array(total_cf.loc[:, cl])
		price = np.sum(cashflows*Z)
		return price

	def calculate_durations_and_convexities(self, total_cf, dr, dt):

		def yield_auxiliary(y, *data):
			cl, dt, price = data
			price_y = self.calculate_price_given_yield(y, total_cf, cl, dt)
			return price_y - price

		y = np.zeros(len(self.classes))
		dur = np.zeros(len(self.classes))
		conv = np.zeros(len(self.classes))
		for cl_ind in range(len(self.classes)):

			cl = self.classes[cl_ind]
			P = self.simulation_summary.loc[cl, 'Average price']

			# Yield
			y[cl_ind] = fsolve(yield_auxiliary, self.pool_interest_rate, args = (cl, dt, P))

			# Prices
			P_up = self.calculate_price_given_yield(y[cl_ind] + dr, total_cf, cl, dt)
			P_dn = self.calculate_price_given_yield(y[cl_ind] - dr, total_cf, cl, dt)

			# Duration
			dur[cl_ind] = (P_dn-P_up)/(P*2*dr)

			# Convexity
			conv[cl_ind] = (P_dn+P_up-2*P)/(P*dr**2)

		dur_conv = np.zeros((3,len(self.classes)))
		dur_conv[0, :] = y
		dur_conv[1, :] = dur
		dur_conv[2, :] = conv

		dur_conv = pd.DataFrame(dur_conv.T, columns = ['Yields', 'Duration', 'Convexity'])
		dur_conv.index = self.classes

		return dur_conv

	def to_minimize_oas(self, spread, par_value, oas_class_cf, monthly_compounded_rates):

		# Discount factors
		Z = 1/(1 + monthly_compounded_rates + spread)
		for i in range(1, Z.shape[1]):
			Z[:, i] = Z[:, i-1]*Z[:, i]

		# Price
		discounted_cf = np.multiply(oas_class_cf, Z)
		price = np.mean(np.sum(discounted_cf, axis=1))

		return (price - par_value)**2

	def calculate_OAS(self, par_value, oas_class_cf, monthly_compounded_rates):
		res =  minimize(self.to_minimize_oas, x0 = 0 , args = (par_value, oas_class_cf, monthly_compounded_rates))
		oas = res.x[0]
		return oas
