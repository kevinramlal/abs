
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fsolve
from datetime import datetime
from utilities import *


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

	def coupon_payment(self, r_month, months_remaining, balance):
		return r_month*balance/(1-1/(1+r_month)**months_remaining)

	def calculate_pool_cf_psa_report(self, PSA):
		'''
			Calculates cashflows for the pools using the PSA reporting standard.
		'''
		columns = ['Total Principal', 'Total Interest', 'Balance', 'Interest Available to CMO']
		self.pool_summary = pd.DataFrame(np.zeros((self.maturity+1, 4)), columns = columns)
		self.pools = []

		for pool_index in range(self.n_pools):
			balance = self.pools_info.loc[pool_index, 'Original Balance']
			r_month = self.pools_info.loc[pool_index, 'WAC']/12/100
			term = self.pools_info.loc[pool_index, 'Term']
			age = self.pools_info.loc[pool_index, 'Age']
			columns = ['PMT', 'Interest', 'Principal', 'CPR', 'SMM', 'Prepay_CF', 'Balance']
			pool = pd.DataFrame(np.zeros((self.maturity+1,7)), columns = columns)
			pool.loc[0,'Balance'] = balance
			for month in range(1, term+1):
				prev_balance = pool.loc[month-1,'Balance']
				pool.loc[month, 'PMT'] = self.coupon_payment(r_month, term - (month - 1), prev_balance)
				pool.loc[month, 'Interest'] = prev_balance*r_month
				pool.loc[month, 'Principal'] = prev_balance if pool.loc[month, 'PMT'] - pool.loc[month, 'Interest'] > prev_balance else pool.loc[month, 'PMT'] - pool.loc[month, 'Interest']
				pool.loc[month, 'CPR'] = 0.06*PSA*min(1, (month + age)/30)
				pool.loc[month, 'SMM'] = 1 - (1 - pool.loc[month, 'CPR'])**(1/12)
				pool.loc[month, 'Prepay_CF'] = pool.loc[month, 'SMM']*(prev_balance - pool.loc[month, 'Principal'])
				pool.loc[month, 'Balance'] = prev_balance - pool.loc[month, 'Principal'] - pool.loc[month, 'Prepay_CF']
			self.pools.append(pool)

		for pool in self.pools:
			self.pool_summary['Total Principal'] += pool['Principal'] + pool['Prepay_CF']
			self.pool_summary['Total Interest'] += pool['Interest']
			self.pool_summary['Balance'] += pool['Balance']

		for month in range(1, self.maturity+1):
			self.pool_summary.loc[month, 'Interest Available to CMO'] = self.pool_interest_rate/12*self.pool_summary.loc[month - 1, 'Balance']

	def calculate_pool_cf_hazard_model(self, hazard_model, simulated_lagged_10_year_rates_A):
		'''
			Has to ran after calling calculate_pool_cf_hazard_model.
			Receives a fitted hazard_model from the Hazard class.
			This function adds relevant columns to self.pool to value the bonds under the hazard_model.
		'''
		# Summer variable
		month_start = self.start_date.month
		t = self.pool_summary.index.values
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

		# Prepayment and cash flows
		prepay_CF = [np.zeros((N, T))]*2
		for pool_index in range(self.n_pools):
			pool = self.pools[pool_index]
			for n in range(1): # N
				cpn_gap_n = cpn_gap[pool_index][n][:T]
				covars = np.array((cpn_gap_n, summer)).T
				smm = hazard_model.calculate_prepayment(t, covars)
				print(smm)
				prev_balance = pool.loc[0, 'Balance']
				for month in range(1, T):
					prepay_CF[pool_index][n][month] = smm[month]*(prev_balance - pool.loc[month, 'Principal'])
					prev_balance = prev_balance - pool.loc[month, 'Principal'] - prepay_CF[pool_index][n][month]

		print(prepay_CF)


	def calculate_pool_groups_proportions(self):
		self.principal_groups_proportions = {}
		total_balance = 0
		for group in self.principal_sequential_pay:
			for cl in self.principal_sequential_pay[group]:
				self.principal_groups_proportions[group] = self.principal_groups_proportions.get(group, 0) + self.classes_info.loc[cl, 'Original Balance']
				total_balance += self.classes_info.loc[cl, 'Original Balance']
		for group in self.principal_groups_proportions:
			self.principal_groups_proportions[group] = float(self.principal_groups_proportions[group])/total_balance

	def calculate_classes_cf(self):
		#calculated cashflow of all bonds given the simulated interest rates
		columns = self.classes
		self.classes_balance = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)
		self.classes_interest = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)
		self.classes_accrued = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)
		self.classes_principal = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)
		self.classes_interest_cf = pd.DataFrame(np.zeros((self.maturity+1, len(columns))), columns = columns)

		# Initial Balance
		for cl in self.classes:
			self.classes_balance.loc[0, cl] = self.classes_info.loc[cl, 'Original Balance']

		for month in range(1, self.pool_summary.shape[0]):

			# Interest
			for cl in self.accrual_classes:
				self.classes_interest.loc[month, cl] = self.pool_interest_rate/12*self.classes_balance.loc[month - 1, cl]

			# Distribute Principal
			principal = self.pool_summary.loc[month, 'Total Principal']
			for group in self.principal_groups_proportions:
				principal_group_remaining = principal*self.principal_groups_proportions[group]
				for cl in self.principal_sequential_pay[group]:
					payment = min(principal_group_remaining, self.classes_balance.loc[month - 1, cl])
					self.classes_principal.loc[month, cl] += payment
					principal_group_remaining -= payment

			# Distribute Interest
			for cl in self.accrual_classes:
				interest_remaining = self.classes_interest.loc[month, cl]
				for cl_prin in self.accruals_sequential_pay[cl]:
					payment = min(interest_remaining, self.classes_balance.loc[month - 1, cl_prin] - self.classes_principal.loc[month, cl_prin])
					self.classes_principal.loc[month, cl_prin] += payment
					interest_remaining -= payment

				last_class = self.accruals_sequential_pay[cl][-1]
				if self.classes_balance.loc[month - 1, last_class] - self.classes_principal.loc[month - 1, last_class] > 0:
					self.classes_principal.loc[month, cl] += interest_remaining
					self.classes_accrued.loc[month, cl] = self.classes_interest.loc[month, cl]
				else:
					self.classes_principal.loc[month, cl] += min(interest_remaining, self.classes_principal.loc[month, last_class])
					self.classes_accrued.loc[month, cl] = min(self.classes_interest.loc[month, cl], self.classes_principal.loc[month, last_class])

			# Update Balance
			for cl in self.regular_classes:
				self.classes_balance.loc[month, cl] = max(0, self.classes_balance.loc[month - 1, cl] + self.classes_accrued.loc[month, cl] - self.classes_principal.loc[month, cl])

			# Interest cash flow
			for cl in self.regular_classes:
				if cl in self.accrual_classes:
					self.classes_interest_cf.loc[month, cl] = self.classes_interest.loc[month, cl] - self.classes_accrued.loc[month, cl]
				else:
					self.classes_interest_cf.loc[month, cl] = self.pool_interest_rate/12*self.classes_balance.loc[month - 1, cl]

		# Total cash flow
		total_interest = self.classes_interest_cf.sum(1)
		self.total_cf = self.classes_principal + self.classes_interest_cf
		coupon_differential = self.pool_summary['Total Principal'] + self.pool_summary['Interest Available to CMO'] - self.total_cf.iloc[:,0:-1].sum(axis=1)
		self.total_cf['R'] = coupon_differential #+ self.total_cf.iloc[:,0:-1].sum(axis=1) * interest



	def price_classes(self, simulated_Z):
		'''
			Calculates prices of all classes given the simulated discount factors.
		'''

		Z_up = simulated_Z[0]
		Z_dn = simulated_Z[1]


		n = Z_up.shape[0]
		simulated_prices = np.zeros((n, len(self.classes)))
		m = self.total_cf.shape[0]

		for i in range(n):
			Z_up_i = np.array(Z_up[i][:m])
			Z_dn_i = np.array(Z_dn[i][:m])

			for cl_ind in range(len(self.classes)):
				cashflows = np.array(self.total_cf.iloc[:, cl_ind])
				simulated_prices[i, cl_ind] = (np.sum(cashflows*Z_up_i) + np.sum(cashflows*Z_dn_i))/2

		summary_np = np.zeros((2,len(self.classes)))
		summary_np[0, :] = simulated_prices.mean(0)
		summary_np[1, :] = simulated_prices.std(0)/np.sqrt(n)

		self.simulation_summary = pd.DataFrame(summary_np.T, columns = ['Average price', 'Standard error'])
		self.simulation_summary.index = self.classes

		if self.show_prints:
			print("\n2a) Monte Carlo results", self.simulation_summary)
			#print(latex_table(simulation_summary, caption = "Simulation summary", label = "2a_summary", index = True))


	def calculate_price_given_yield(self, y, cl, dt):
		'''
			Continuous compounding.
			cl is the class
		'''
		m = self.total_cf.shape[0]
		Z = np.exp(-y*np.arange(1, m+1)*dt)
		cashflows = np.array(self.total_cf.loc[:, cl])
		price = np.sum(cashflows*Z)
		return price

	def calculate_durations_and_convexities(self, dr, dt):

		def yield_auxiliary(y, *data):
			cl, dt, price = data
			price_y = self.calculate_price_given_yield(y, cl, dt)
			return price_y - price

		y = np.zeros(len(self.classes))
		dur = np.zeros(len(self.classes))
		conv = np.zeros(len(self.classes))
		for cl_ind in range(len(self.classes)):

			cl = self.classes[cl_ind]
			P = self.simulation_summary.loc[cl, 'Average price']

			# Yield
			y[cl_ind] = fsolve(yield_auxiliary, 0.05, args = (cl, dt, P))

			# Prices
			P_up = self.calculate_price_given_yield(y[cl_ind] + dr, cl, dt)
			P_dn = self.calculate_price_given_yield(y[cl_ind] - dr, cl, dt)

			# Duration
			dur[cl_ind] = (P_dn-P_up)/(P*2*dr)

			# Convexity
			conv[cl_ind] = (P_dn+P_up-2*P)/(P*dr**2)

		dur_conv = np.zeros((3,len(self.classes)))
		dur_conv[0, :] = y
		dur_conv[1, :] = dur
		dur_conv[2, :] = conv

		self.dur_conv = pd.DataFrame(dur_conv.T, columns = ['Yields', 'Duration', 'Convexity'])
		self.dur_conv.index = self.classes

		if self.show_prints:
			print("\n2b) Duration and convexity results", self.simulation_summary)
			#print(latex_table(self.dur_conv, caption = "Duration and Convexity", label = "2b_summary", index = True))

		return (dur, conv)

	def to_minimize_oas(self, params, sim_avg_rates, cfs,par):
		oas = params[0]
		res = 0
		dts = np.array([ i*1/12 for i in range(len(cfs))])
		dcfs = np.multiply(np.exp(-1*np.multiply(sim_avg_rates[:len(cfs)] + oas, dts)), cfs)
		res = np.abs(par- np.sum(dcfs))
		return res

	def find_oas_classes(self, simulated_rates_avg):
		oas_summary_np = summary_np = np.zeros((1,len(self.classes)))
		for cl_ind in range(len(self.classes)):
			oas = 0.0
			cashflows = np.array(self.total_cf.iloc[:, cl_ind])
			par = self.classes_balance.iloc[0, cl_ind]
			optimum = minimize(self.to_minimize_oas, x0 = [oas] , args = (simulated_rates_avg, cashflows,par))
			oas_summary_np[0, cl_ind] = optimum.x
		oas_summary = pd.DataFrame(oas_summary_np, columns=self.classes)
		oas_summary.index = ['OAS']
		oas_summary = oas_summary.drop(columns=['R'])

		if self.show_prints:
			print("\n2c) OAS results", oas_summary)

