
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

	def __init__(self, pool_origination_date, evaluation_date, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay, simulated_rates, tables_file, show_prints=False, show_plots=False):
		# Direct inputs
		self.pool_origination_date = datetime.strptime(pool_origination_date, "%m/%d/%Y")
		self.evaluation_date = datetime.strptime(evaluation_date, "%m/%d/%Y")
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
		self.evaluation_lag = (12*self.evaluation_date.year + self.evaluation_date.month) - (12*self.pool_origination_date.year + self.pool_origination_date.month)
		self.n_pools = self.pools_info.shape[0]
		self.classes = list(self.classes_info['REMIC Classes'])
		self.classes_ordered = principal_sequential_pay['1']
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

		simulation_summary = self.calculate_price(hz_frm_prepay, hz_frm_default, hz_arm_prepay, hz_arm_default, simulated_lagged_10_year_rates_A, dr=0)

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
		self.calculate_cashflows(hz_frm_prepay, hz_frm_default, hz_arm_prepay, hz_arm_default)
		self.plot_results()

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
		#	pool_summary_n = self.calculate_cashflows(SMM_n, r_n)
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
		month_start = self.evaluation_date.month
		t = np.arange(0,self.T) + self.evaluation_lag
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

	def coupon_payment(self, r_month, months_remaining, balance):
		return r_month*balance/(1-1/(1+r_month)**months_remaining)

	def calculate_cashflows(self, hz_frm_prepay, hz_frm_default, hz_arm_prepay, hz_arm_default):
		'''
			Calculates pool cashflows considering prepayment and defaults of FRM and ARM pools.
			Then distributes interest and principal to bond classes.
			Defaults is managed using excess spread first and then the overcollateralization amount.
			Positive excess spread is saved in the overcollateralization amount until a target is reached, at which point it starts to be distributed as principal.
		'''

		# Contractual
		r_month_frm = self.pool_frm_wac/12/100
		r_month_arm = self.ARM_simulated_rates_capped/12/100
		self.balance_frm = np.zeros((self.N, self.T))
		self.balance_arm = np.zeros((self.N, self.T))
		amortization_pmt_frm = np.zeros((self.N, self.T))
		amortization_pmt_arm = np.zeros((self.N, self.T))
		self.interest_pmt_frm = np.zeros((self.N, self.T))
		self.interest_pmt_arm = np.zeros((self.N, self.T))
		self.principal_pmt_frm = np.zeros((self.N, self.T))
		self.principal_pmt_arm = np.zeros((self.N, self.T))

		# Prepayment
		self.principal_prepay_frm = np.zeros((self.N, self.T))
		self.principal_prepay_arm = np.zeros((self.N, self.T))

		# Default
		ltv_frm = np.zeros((self.N, self.T))
		ltv_arm = np.zeros((self.N, self.T))
		self.default_frm = np.zeros((self.N, self.T))
		self.default_arm = np.zeros((self.N, self.T))
		self.principal_default_frm = np.zeros((self.N, self.T))
		self.principal_default_arm = np.zeros((self.N, self.T))
		self.balance_frm[:, 0] = self.pool_frm_balance
		self.balance_arm[:, 0] = self.pool_arm_balance
		ltv_frm[:, 0] = self.pool_frm_ltv
		ltv_arm[:, 0] = self.pool_arm_ltv

		# Default management
		overcollateralization = np.zeros((self.N, self.T))

		# Bonds
		self.bonds_balance = {}
		self.bonds_interest = {}
		self.bonds_principal = {}
		self.bonds_extra_principal = {} # Distributed due to excess spread after overcollateralization target has been achieved.
		self.bonds_principal_default = {}
		self.overcollateralization_amount = np.zeros((self.N, self.T)) # We will assume it starts at zero, given we do not have this information
		for cl in self.classes_ordered:
			self.bonds_balance[cl] = np.zeros((self.N, self.T))
			self.bonds_balance[cl][:, 0] = self.classes_info.loc[cl,'Balance']
			self.bonds_interest[cl] = np.zeros((self.N, self.T))
			self.bonds_principal[cl] = np.zeros((self.N, self.T))
			self.bonds_extra_principal[cl] = np.zeros((self.N, self.T))
			self.bonds_principal_default[cl] = np.zeros((self.N, self.T))

		for month in range(1, self.T):

			# ----------------------------------
			#  Pool cash flows
			# ----------------------------------

			# Contratcual
			prev_balance_frm = self.balance_frm[:, month-1]
			prev_balance_arm = self.balance_arm[:, month-1]
			# Mortgage owners pay equal monthly payments. We will call that amortization payment (also called coupon payment)
			# The amortization payment has to be recalculated every period because of prepayment
			amortization_pmt_frm[:, month] = self.coupon_payment(r_month_frm, self.T - month , prev_balance_frm)
			amortization_pmt_arm[:, month] = self.coupon_payment(r_month_arm[:, month-1], self.T - month, prev_balance_arm)
			self.interest_pmt_frm[:, month] = prev_balance_frm*r_month_frm
			self.interest_pmt_arm[:, month] = prev_balance_arm*r_month_arm[:, month-1]
			self.principal_pmt_frm[:, month] = amortization_pmt_frm[:, month] - self.interest_pmt_frm[:, month]
			self.principal_pmt_arm[:, month] = amortization_pmt_arm[:, month] - self.interest_pmt_arm[:, month]
			# We manage explicitly an extreme case that may happen in the last payment because of rounding error
			extreme_case_frm = amortization_pmt_frm[:, month] - self.interest_pmt_frm[:, month] > prev_balance_frm
			extreme_case_arm = amortization_pmt_arm[:, month] - self.interest_pmt_arm[:, month] > prev_balance_arm
			self.principal_pmt_frm[extreme_case_frm, month] = prev_balance_frm[extreme_case_frm]
			self.principal_pmt_arm[extreme_case_arm, month] = prev_balance_arm[extreme_case_arm]

			# Prepayment
			self.principal_prepay_frm[:, month] = self.SMM_frm[:, month]*(prev_balance_frm - self.principal_pmt_frm[:, month])
			self.principal_prepay_arm[:, month] = self.SMM_arm[:, month]*(prev_balance_arm - self.principal_pmt_arm[:, month])

			# Default
			house_increase_factor = self.hp_frm[:, month]/self.hp_frm[:, month-1]
			ltv_frm[:, month] = ltv_frm[:, month-1]/house_increase_factor
			ltv_arm[:, month] = ltv_arm[:, month-1]/house_increase_factor
			ltv_frm_month = np.resize(ltv_frm[:, month], (self.N, 1))
			ltv_arm_month = np.resize(ltv_arm[:, month], (self.N, 1))
			self.default_frm[:, month] = hz_frm_default.calculate_default(month + self.evaluation_lag, ltv_frm_month)
			self.default_arm[:, month] = hz_arm_default.calculate_default(month + self.evaluation_lag, ltv_arm_month)
			principal_default_frm_total = self.default_frm[:, month]*(prev_balance_frm - self.principal_pmt_frm[:, month])
			principal_default_arm_total = self.default_arm[:, month]*(prev_balance_arm - self.principal_pmt_arm[:, month])
			self.principal_default_frm[:, month] = principal_default_frm_total*0.6
			self.principal_default_arm[:, month] = principal_default_arm_total*0.6
			self.principal_prepay_frm[:, month] += principal_default_frm_total*0.4 # Recovery
			self.principal_prepay_arm[:, month] += principal_default_arm_total*0.4
			self.balance_frm[:, month] = prev_balance_frm - self.principal_pmt_frm[:, month] - self.principal_prepay_frm[:, month] - self.principal_default_frm[:, month]
			self.balance_arm[:, month] = prev_balance_arm - self.principal_pmt_arm[:, month] - self.principal_prepay_arm[:, month] - self.principal_default_arm[:, month]

			# Interest and Principal distribution
			remaining_interest = self.interest_pmt_frm[:, month] + self.interest_pmt_arm[:, month]
			remaining_principal = self.principal_pmt_frm[:, month] + self.principal_pmt_arm[:, month] + self.principal_prepay_frm[:, month] + self.principal_prepay_arm[:, month]
			for cl in self.classes_ordered:
				# Interest
				prev_balance = self.bonds_balance[cl][:, month-1]
				interest_rate = (self.simulated_rates_APR[:, month-1] + self.classes_info.loc[cl,'Spread'])/12/100
				interest_cl = prev_balance*interest_rate
				self.bonds_interest[cl][:, month] = np.minimum(interest_cl, remaining_interest)
				remaining_interest -= np.minimum(interest_cl, remaining_interest)

				# Principal
				principal_cl = np.minimum(prev_balance, remaining_principal)
				self.bonds_principal[cl][:, month] = principal_cl
				self.bonds_balance[cl][:, month] = self.bonds_balance[cl][:, month-1] - principal_cl
				remaining_principal -= principal_cl
			excess_spread = remaining_interest

			# ----------------------------------
			#  Default management
			# ----------------------------------

			# If there is any cash in the overcollateralization accounts, then defaults hits there first.
			# Then, excess spread nets as much of the remaining default as possible.
			# IFFFF there happens to be excess spread still, then it is stored in the overcollateralization account.
			# This is, to the extent that this account is capped by the Target.
			# Any excess to the Target is distributed to the R class that is not studied here.
			# All overcollateralization and excess spread matched with default is paid inmediately following the priority sequence.

			principal_default = self.principal_default_frm[:, month] + self.principal_default_arm[:, month]

			# Step 1: Default is paid by the overcollateralization account as much as possible.
			overcollateralization_pmt = np.minimum(self.overcollateralization_amount[:, month-1], principal_default)
			principal_default_remaining = principal_default - overcollateralization_pmt
			self.overcollateralization_amount[:, month] = self.overcollateralization_amount[:, month-1] - overcollateralization_pmt

			# Step 2: Remaining default is paid by excess spread as much as possible.
			excess_spread_pmt = np.minimum(excess_spread, principal_default_remaining)
			principal_default_remaining = principal_default_remaining - excess_spread_pmt
			extra_excess_spread = excess_spread - excess_spread_pmt

			# Step 3: If there was extra excess spread, it is saved as cash in the overcollateralization account capped by a predefined target.
			option_1 = np.zeros(self.N) + 0.031*(79036000 + 714395000)
			option_2 = 0.062*(self.balance_frm[:, month] + self.balance_arm[:, month])
			option_3 = 3967158
			overcollateralization_target = np.maximum(option_1, np.maximum(option_2, option_3))
			self.overcollateralization_amount[:, month] = np.minimum(self.overcollateralization_amount[:, month] + extra_excess_spread, overcollateralization_target)

			# Step 4: Distribute overcollateralization and excess spread matched with default
			extra_principal = overcollateralization_pmt + excess_spread_pmt
			for cl in self.classes_ordered:
				balance_cl = self.bonds_balance[cl][:, month]
				principal_cl = np.minimum(balance_cl, extra_principal)
				self.bonds_principal[cl][:, month] += principal_cl
				self.bonds_balance[cl][:, month] -= principal_cl
				extra_principal -= principal_cl

			# Step 5: Assign remaining defaults
			for cl in self.classes_ordered[::-1]:
				balance_cl = self.bonds_balance[cl][:, month]
				principal_cl = np.minimum(balance_cl, principal_default_remaining)
				self.bonds_principal_default[cl][:, month] = principal_cl
				self.bonds_balance[cl][:, month] -= principal_cl
				principal_default_remaining -= principal_cl


	def plot_results(self):

		# -------------------------------
		#  Pools
		# -------------------------------

		# Prepayment
		t0 = np.arange(0, self.T)
		evaluation_date = str(self.evaluation_date.date())
		plt.plot(t0, self.SMM_frm.mean(axis=0)*100, label="FRM")
		plt.plot(t0, self.SMM_arm.mean(axis=0)*100, label="ARM")
		plt.ylabel("Average SMM (%)")
		plt.xlabel("Months from "+ evaluation_date)
		plt.legend()
		plt.show()

		# Default (%)
		t1 = np.arange(1, self.T)
		plt.plot(t1, self.default_frm[:, 1:].mean(axis=0)*100, label="FRM")
		plt.plot(t1, self.default_arm[:, 1:].mean(axis=0)*100, label="ARM")
		plt.ylabel("Average default (%)")
		plt.xlabel("Months from "+ evaluation_date)
		plt.legend()
		plt.show()

		# Balance
		plt.plot(t0, self.balance_frm.mean(axis=0)/1e6, label="FRM")
		plt.plot(t0, self.balance_arm.mean(axis=0)/1e6, label="ARM")
		plt.ylabel("Average balance ($mm)")
		plt.xlabel("Months from "+ evaluation_date)
		plt.legend()
		plt.show()

		# Interest payment
		plt.plot(t0, self.interest_pmt_frm.mean(axis=0)/1e6, label="FRM")
		plt.plot(t0, self.interest_pmt_arm.mean(axis=0)/1e6, label="ARM")
		plt.ylabel("Average interest payment ($mm)")
		plt.xlabel("Months from "+ evaluation_date)
		plt.legend()
		plt.show()

		# Principal payment (including prepayment)
		plt.plot(t0, self.principal_pmt_frm.mean(axis=0)/1e6, label="FRM")
		plt.plot(t0, self.principal_pmt_arm.mean(axis=0)/1e6, label="ARM")
		plt.ylabel("Average scheduled principal ($mm)")
		plt.xlabel("Months from "+ evaluation_date)
		plt.legend()
		plt.show()

		# Principal prepayment
		plt.plot(t0, self.principal_prepay_frm.mean(axis=0)/1e6, label="FRM")
		plt.plot(t0, self.principal_prepay_arm.mean(axis=0)/1e6, label="ARM")
		plt.ylabel("Average principal prepayment ($mm)")
		plt.xlabel("Months from "+ evaluation_date)
		plt.legend()
		plt.show()

		# Default losses
		plt.plot(t0, self.principal_default_frm.mean(axis=0)/1e6, label="FRM")
		plt.plot(t0, self.principal_default_arm.mean(axis=0)/1e6, label="ARM")
		plt.ylabel("Average default losses ($mm)")
		plt.xlabel("Months from "+ evaluation_date)
		plt.legend()
		plt.show()


		# -------------------------------
		#  Bonds
		# -------------------------------

		# Balance
		for cl in self.classes_ordered:
			plt.plot(t0, self.bonds_balance[cl].mean(axis=0), label=cl)
		plt.ylabel("Average balance ($)")
		plt.xlabel("Months from "+ evaluation_date)
		plt.legend()
		plt.show()

		# Overcollateralization amount
		plt.plot(t0, self.overcollateralization_amount.mean(axis=0))
		plt.ylabel("Average overcollateralization amount ($)")
		plt.xlabel("Months from "+ evaluation_date)
		plt.show()























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
