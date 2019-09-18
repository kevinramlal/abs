
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve
from datetime import datetime
import time
import sys
import fixed_income
from utilities import *

import warnings
warnings.filterwarnings("ignore")

class REMIC:
	'''
		Values REMIC bonds and calculates relevant metrics.
	'''

	def __init__(self, start_date, first_payment_date, pool_interest_rate, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay, simulated_rates, tables_file, show_prints=False, show_plots=False):
		# Direct inputs
		self.start_date = datetime.strptime(start_date, "%m/%d/%Y")
		self.first_payment_date = datetime.strptime(first_payment_date, "%m/%d/%Y")
		self.pool_interest_rate = pool_interest_rate
		self.pools_info = pools_info
		self.classes_info = classes_info
		self.principal_sequential_pay = principal_sequential_pay
		self.accruals_sequential_pay = accruals_sequential_pay
		self.simulated_rates_cont = simulated_rates
		self.simulated_rates_APR = 12*(np.exp(simulated_rates*1/12)-1)*100 # Monthly APR in %
		self.show_prints = show_prints
		self.show_plots = show_plots
		self.tables_file = tables_file
		self.fi = fixed_income.FixedIncome()

		# Processed attributes
		self.maturity = np.max(self.pools_info['Term'])
		self.T = min(self.maturity+1, simulated_rates.shape[1]) # Months remaining counting month 0
		self.N = simulated_rates.shape[0] # Number of simulations
		self.n_pools = self.pools_info.shape[0]
		self.classes = list(self.classes_info['REMIC Classes'])
		self.regular_classes = self.classes[:]
		if 'R' in self.regular_classes:
			self.regular_classes.remove('R')
		self.accrual_classes = list(self.accruals_sequential_pay.keys())
		self.principal_classes = list(set(self.regular_classes) - set(self.accrual_classes))

		self.pools_info.index = self.pools_info['Pool ID']
		self.pools_info['Balance'] = self.pools_info['Balance'].astype(float)
		self.pools_info['WAC'] = self.pools_info['WAC'].astype(float)
		self.pools_info['Spread'] = self.pools_info['Spread'].astype(float)
		self.pools_info['LOL Cap'] = self.pools_info['LOL Cap'].astype(float)
		self.pools_info['Periodic Cap'] = self.pools_info['Periodic Cap'].astype(float)
		self.pools_info['LTV'] = self.pools_info['LTV'].astype(float)

		self.pool_frm_balance = self.pools_info.loc['FRM','Balance']
		self.pool_frm_wac = self.pools_info.loc['FRM','WAC']
		self.pool_frm_ltv = self.pools_info.loc['FRM','LTV']

		self.pool_arm_balance = self.pools_info.loc['ARM','Balance']
		self.pool_arm_spread = self.pools_info.loc['ARM','Spread']
		self.pool_arm_lol_cap = self.pools_info.loc['ARM','LOL Cap']
		self.pool_arm_periodic_cap = self.pools_info.loc['ARM','Periodic Cap']
		self.pool_arm_ltv = self.pools_info.loc['ARM','LTV']

		self.classes_info.index = self.classes_info['REMIC Classes']
		self.classes_info['Balance'] = self.classes_info['Balance'].astype(float)
		self.classes_info['Spread'] = self.classes_info['Spread'].astype(float)


		self.calculate_ARM_simulated_rates()

	def simulation_result(self, hz_frm_prepay, hz_frm_default, hz_arm_prepay, hz_arm_default, simulated_lagged_10_year_rates_A):

		#---------------------------
		# Bonds
		#---------------------------

		simulation_summary = self.calculate_price(hz_frm_prepay, hz_frm_default, hz_arm_prepay, hz_frm_default, simulated_lagged_10_year_rates_A, dr=0)

		#if self.show_prints:
			#print('\nPart '+ part_price + ':\n' + str(simulation_summary) + '\n\n')
			#self.tables_file.write(latex_table(simulation_summary, caption = "Simulation summary", label = "prices", index = True))

		#---------------------------
		# CDS
		#---------------------------



	def calculate_price(self, hz_frm_prepay, hz_frm_default, hz_arm_prepay, hz_arm_default, simulated_lagged_10_year_rates_A, dr=0):
		'''
			Calculates the average simulated price.
			Allows parallel shocks in interest rates with dr.
		'''

		self.calculate_pool_simulation_prepayment(hz_frm_prepay, hz_arm_prepay, simulated_lagged_10_year_rates_A + dr)
		#self.calculate_pool_simulation_default(hz_frm_default, hz_arm_default, frm_remaining_bal, arm_remaining_bal)
		self.calculate_pool_cf(hz_frm_prepay, hz_frm_default, hz_arm_prepay, hz_arm_default)

		plt.plot(self.SMM_frm[0], label="FRM")
		plt.plot(self.SMM_arm[0], label="ARM")
		plt.legend()
		plt.show()

		#############################################################################################################################

		# Sagnik, try to implement these two functions
		# house_prices = simulate_house_prices(...)
		#(Default_frm, Default_arm) = self.calculate_pool_simulation_default(hz_frm_default, hz_arm_default, house_prices, ...)

		#############################################################################################################################


		#N = simulated_lagged_10_year_rates_A.shape[0]
		#Nh = int(N/2)
		#prices_all = np.zeros((N, len(self.classes)))
		#prices_paired = np.zeros((Nh, len(self.classes)))
#
		#r = self.simulated_rates + dr
		#Z = self.fi.hull_white_discount_factors_antithetic_path(r, dt=1/12)
#
		#print("Calculating cash flows for each path...")
		#for n in range(N):
		#	SMM_n = [SMM[i][n] for i in range(self.n_pools)]
		#	r_n = r[n]
		#	Z_n = Z[n]
		#	pool_summary_n = self.calculate_pool_cf(SMM_n, r_n)
		#	total_cf = self.calculate_classes_cf(pool_summary_n, r_n)
		#	prices_all[n] = self.price_classes(total_cf, Z_n)
#
		## Pair antithetic prices
		#for n in range(Nh):
		#	prices_paired[n] = (prices_all[n] + prices_all[n+Nh])/2
#
		#summary_np = np.zeros((3,len(self.classes)))
		#summary_np[0, :] = prices_paired.mean(axis=0)
		#summary_np[1, :] = prices_paired.std(axis=0)
		#summary_np[2, :] = summary_np[1, :]/np.sqrt(Nh)
#
		#simulation_summary = pd.DataFrame(summary_np.T, columns = ['Average Price', 'Std. Deviation', 'Std. Error'])
		#simulation_summary.index = self.classes
#
		#return simulation_summary


	def simulate_house_prices(self, n, q, phi):
		'''
			Simulates n paths of house prices.
			Returns two numpy arrays one for FRM and one for ARM.
			Rows indicate simulation path and columns indicate month.

			What do we need for this? We know dH = (r-q) * H * dt + phi * H * dW.
			So we need, r which is the riskless short rate, q and phi which are constants, dt which we know.
		'''
		dt = 1/12
		h0_frm = self.pool_frm_balance/self.pool_frm_ltv
		h0_arm = self.pool_arm_balance/self.pool_arm_ltv

		## House price simulations
		np.random.seed(0)
		self.hp_frm = np.zeros((self.N, self.T))
		self.hp_arm = np.zeros((self.N, self.T))

		self.hp_frm[:, 0] = h0_frm
		self.hp_arm[:, 0] = h0_arm

		for i in range(1, self.T):
			w = np.random.normal(0, 1, self.N)
			dh_frm = (self.simulated_rates_cont[:,i] - q)*self.hp_frm[:, i-1]*dt + phi*self.hp_frm[:, i-1]*w*np.sqrt(dt)
			dh_arm = (self.simulated_rates_cont[:,i] - q)*self.hp_arm[:, i-1]*dt + phi*self.hp_frm[:, i-1]*w*np.sqrt(dt)
			self.hp_frm[:, i] = self.hp_frm[:, i-1] + dh_frm
			self.hp_arm[:, i] = self.hp_arm[:, i-1] + dh_arm

	def calculate_ARM_simulated_rates(self):

		T = self.simulated_rates_APR.shape[1]
		ARM_simulated_rates = self.simulated_rates_APR + self.pool_arm_spread

		self.ARM_simulated_rates_capped = np.copy(ARM_simulated_rates)

		# Periodic cap
		for i in range(1, T):
			dr = ARM_simulated_rates[:, i] - self.ARM_simulated_rates_capped[:, i-1]
			bool_periodic_cap_pos = dr > self.pool_arm_periodic_cap
			bool_periodic_cap_neg = dr < -self.pool_arm_periodic_cap
			self.ARM_simulated_rates_capped[bool_periodic_cap_pos, i] = self.ARM_simulated_rates_capped[bool_periodic_cap_pos, i-1] + self.pool_arm_periodic_cap
			self.ARM_simulated_rates_capped[bool_periodic_cap_neg, i] = self.ARM_simulated_rates_capped[bool_periodic_cap_neg, i-1] - self.pool_arm_periodic_cap

		# LOL cap
		bool_lol_cap = self.ARM_simulated_rates_capped > self.pool_arm_lol_cap
		self.ARM_simulated_rates_capped[bool_lol_cap] = self.pool_arm_lol_cap


	def calculate_pool_simulation_prepayment(self, hz_frm_prepay, hz_arm_prepay, simulated_lagged_10_year_rates_A):
		'''
			Receives a fitted hazard_model from the Hazard class.
			Returns numpy array with SMM where rows indicate simulation path and columns indicate month.
		'''
		# Summer variable
		month_start = self.start_date.month
		t = np.arange(0,self.T)
		month_index = np.mod(t + month_start - 1, 12) + 1
		summer = np.array([1.0 if i>=5 and i<=8 else 0.0 for i in month_index])

		# Coupon gap
		cpn_gap = []
		# cpn_gap is a list of numpy arrays containing the coupon gap for every simulation and month.
		# The list has one numpy array for each pool.
		# Every row indicates a simulation and every column a month.
		cpn_gap_frm = self.pool_frm_wac - simulated_lagged_10_year_rates_A[:,:self.T]*100
		cpn_gap_arm = self.ARM_simulated_rates_capped[:,:self.T] - simulated_lagged_10_year_rates_A[:,:self.T]*100

		## Prepayment
		self.SMM_frm = np.zeros((self.N, self.T))
		self.SMM_arm = np.zeros((self.N, self.T))
		for n in range(self.N):
			cpn_gap_frm_n = cpn_gap_frm[n]
			cpn_gap_arm_n = cpn_gap_arm[n]
			covars_frm = np.array((cpn_gap_frm_n, summer)).T
			covars_arm = np.array((cpn_gap_arm_n, summer)).T
			self.SMM_frm[n] = hz_frm_prepay.calculate_prepayment(t, covars_frm)
			self.SMM_arm[n] = hz_arm_prepay.calculate_prepayment(t, covars_arm)

	def calculate_pool_cf(self, hz_frm_prepay, hz_frm_default, hz_arm_prepay, hz_arm_default):
		'''
			Calculates pool cashflows considering prepayment and defaults of FRM and ARM pools.
		'''
		r_month_frm = self.pool_frm_wac/12/100
		r_month_arm = self.ARM_simulated_rates_capped/12/100
		balance_frm = np.zeros((self.N, self.T))
		balance_arm = np.zeros((self.N, self.T))
		amortization_pmt_frm = np.zeros((self.N, self.T))
		amortization_pmt_arm = np.zeros((self.N, self.T))
		interest_pmt_frm = np.zeros((self.N, self.T))
		interest_pmt_arm = np.zeros((self.N, self.T))
		principal_pmt_frm = np.zeros((self.N, self.T))
		principal_pmt_arm = np.zeros((self.N, self.T))
		principal_prepay_frm = np.zeros((self.N, self.T))
		principal_prepay_arm = np.zeros((self.N, self.T))
		ltv_frm = np.zeros((self.N, self.T))
		ltv_arm = np.zeros((self.N, self.T))
		self.default_frm = np.zeros((self.N, self.T))
		self.default_arm = np.zeros((self.N, self.T))
		principal_default_frm = np.zeros((self.N, self.T))

		balance_frm[:, 0] = self.pool_frm_balance
		balance_arm[:, 0] = self.pool_arm_balance
		ltv_frm[:, 0] = self.pool_frm_ltv
		ltv_arm[:, 0] = self.pool_arm_ltv
        
		for month in range(1, self.T):

			# Contratcual cash flows
			prev_balance_frm = balance_frm[:, month-1]
			prev_balance_arm = balance_arm[:, month-1]
			# Mortgage owners pay equal monthly payments. We will call that amortization payment (also called coupon payment)
			# The amortization payment has to be recalculated every period because of prepayment
			amortization_pmt_frm[:, month] = self.coupon_payment(r_month_frm, self.T - month , prev_balance_frm)
			amortization_pmt_arm[:, month] = self.coupon_payment(r_month_arm[:, month-1], self.T - month, prev_balance_arm)
			interest_pmt_frm[:, month] = prev_balance_frm*r_month_frm
			interest_pmt_arm[:, month] = prev_balance_arm*r_month_arm[:, month-1]
			principal_pmt_frm[:, month] = amortization_pmt_frm[:, month] - interest_pmt_frm[:, month]
			principal_pmt_arm[:, month] = amortization_pmt_arm[:, month] - interest_pmt_arm[:, month]
			# We manage explicitly an extreme case that may happen in the last payment because of rounding error
			extreme_case_frm = amortization_pmt_frm[:, month] - interest_pmt_frm[:, month] > prev_balance_frm
			extreme_case_arm = amortization_pmt_arm[:, month] - interest_pmt_arm[:, month] > prev_balance_arm
			principal_pmt_frm[extreme_case_frm, month] = prev_balance_frm[extreme_case_frm]
			principal_pmt_arm[extreme_case_arm, month] = prev_balance_arm[extreme_case_arm]

			# Prepayment
			principal_prepay_frm[:, month] = self.SMM_frm[:, month]*(prev_balance_frm - principal_pmt_frm[:, month])
			principal_prepay_arm[:, month] = self.SMM_arm[:, month]*(prev_balance_arm - principal_pmt_arm[:, month])
			remaining_balance_frm = prev_balance_frm - principal_pmt_frm[:, month] - principal_prepay_frm[:, month]
			remaining_balance_arm = prev_balance_arm - principal_pmt_arm[:, month] - principal_prepay_arm[:, month]

			# Default
			house_increase_factor = self.hp_frm[:, month]/self.hp_frm[:, month-1]
			ltv_frm[:, month] = ltv_frm[:, month-1]/house_increase_factor
			ltv_arm[:, month] = ltv_arm[:, month-1]/house_increase_factor
			ltv_frm_month = np.resize(ltv_frm[:, month], (self.N, 1))
			ltv_arm_month = np.resize(ltv_arm[:, month], (self.N, 1))
			self.default_frm[:, month] = hz_frm_default.calculate_default(month, ltv_frm_month)
			self.default_arm[:, month] = hz_arm_default.calculate_default(month, ltv_arm_month)
			principal_default_frm = self.default_frm[:, month]*remaining_balance_frm
			principal_default_arm = self.default_arm[:, month]*remaining_balance_arm
			balance_frm[:, month] = remaining_balance_frm - principal_default_frm
			balance_arm[:, month] = remaining_balance_arm - principal_default_arm


		self.total_principal_pmt = principal_pmt_frm + principal_pmt_arm + principal_prepay_frm + principal_prepay_arm
		self.total_interest_pmt = interest_pmt_frm + interest_pmt_arm
		self.total_principal_default = principal_default_frm + principal_default_arm
		self.total_balance = balance_frm + balance_arm


		print("balance_frm", balance_frm)
		print("balance_arm", balance_arm)

	def coupon_payment(self, r_month, months_remaining, balance):
		return r_month*balance/(1-1/(1+r_month)**months_remaining)





























	def calculate_pool_cf_old(self, SMM):
		'''
			Calculates pool cashflows given monthly prepayment (SMM)
		'''
		columns = ['Total Principal', 'Total Interest', 'Balance', 'Interest Available to CMO']
		pool_summary = pd.DataFrame(np.zeros((self.maturity+1, 4)), columns = columns)
		pools = []

		for pool_index in range(self.n_pools):
			balance = self.pools_info.loc[pool_index, 'Balance']
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
			classes_balance.loc[0, cl] = self.classes_info.loc[cl, 'Balance']

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
		total_cf['R'] = coupon_differential + total_cf.iloc[:,0:-1].sum(axis=1)*((1+r[:total_cf.shape[0]]/12)**(0.5)-1)

		return total_cf.iloc[1:,:]

	def price_classes(self, total_cf, Z):

		prices = np.zeros(len(self.classes))
		m = total_cf.shape[0]
		for cl_ind in range(len(self.classes)):
			cashflows = np.array(total_cf.iloc[:, cl_ind])
			prices[cl_ind] = np.sum(cashflows*Z[:m])
		return prices

	def calculate_durations_and_convexities(self, simulation_summary, hazard_model,  simulated_lagged_10_year_rates_A, dr, dt):

		simulation_summary_up = self.calculate_price(hazard_model, simulated_lagged_10_year_rates_A, dr=dr)
		simulation_summary_dn = self.calculate_price(hazard_model, simulated_lagged_10_year_rates_A, dr=-dr)

		dur = np.zeros(len(self.classes))
		conv = np.zeros(len(self.classes))

		P = simulation_summary['Average Price']
		P_up = simulation_summary_up['Average Price']
		P_dn = simulation_summary_dn['Average Price']

		dur = (P_dn-P_up)/(P*2*dr)
		conv = (P_dn+P_up-2*P)/(P*dr**2)

		dur_conv = np.zeros((2,len(self.classes)))
		dur_conv[0, :] = dur
		dur_conv[1, :] = conv

		dur_conv = pd.DataFrame(dur_conv.T, columns = ['Duration', 'Convexity'])
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

	def calculate_OAS(self, oas_class, hazard_model, simulated_lagged_10_year_rates_A):

		T = self.maturity
		N = simulated_lagged_10_year_rates_A.shape[0]
		Nh = int(N/2)
		dt = 1/12
		par_value = self.classes_info.loc[oas_class, 'Balance']
		# Spot rates are continously compounded.
		# Accoding to the lectures, OAS is the spread using monthly compounding.
		monthly_compounded_rates = np.exp(self.simulated_rates[:Nh, :T]*dt)-1

		SMM = self.calculate_pool_simulation_prepayment(hazard_model, simulated_lagged_10_year_rates_A)
		r = self.simulated_rates[:Nh]
		Z = self.fi.hull_white_discount_factors_antithetic_path(r, dt=1/12)
		oas_class_cf = np.zeros((Nh, self.maturity))

		print("Calculating cash flows for each path...")
		for n in range(Nh):
			SMM_n = [SMM[i][n] for i in range(self.n_pools)]
			r_n = r[n]
			Z_n = Z[n]
			pool_summary_n = self.calculate_pool_cf(SMM_n)
			total_cf = self.calculate_classes_cf(pool_summary_n, r_n)
			oas_class_cf[n] = total_cf[oas_class]

		res =  minimize(self.to_minimize_oas, x0 = 0 , args = (par_value, oas_class_cf, monthly_compounded_rates))
		oas = res.x[0]
		return oas
