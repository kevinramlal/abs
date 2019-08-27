
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 9)
pd.set_option('display.max_rows', 238)

class REMIC:

	def __init__(self, today, first_payment_date, pool_interest_rate, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay):
		# Direct inputs
		self.today = today
		self.first_payment_date = first_payment_date
		self.pool_interest_rate = pool_interest_rate
		self.pools_info = pools_info
		self.classes_info = classes_info
		self.principal_sequential_pay = principal_sequential_pay
		self.accruals_sequential_pay = accruals_sequential_pay

		# Processed attributes
		self.maturity = np.max(self.pools_info['Term'])
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

	def calculate_pool_cf(self, PSA):
		columns = ['Total Principal', 'Total Interest', 'Balance', 'Interest Available to CMO']
		self.pool_summary = pd.DataFrame(np.zeros((self.maturity+1, 4)), columns = columns)
		pools = []

		for pool_index in range(self.pools_info.shape[0]):
			balance = self.pools_info.loc[pool_index, 'Original Balance']
			r_month = self.pools_info.loc[pool_index, 'WAC']/12/100
			term = self.pools_info.loc[pool_index, 'Term']
			age = self.pools_info.loc[pool_index, 'Age']
			columns = ['PMT', 'Interest', 'Principal', 'CPR', 'SMM', 'Prepay CF', 'Balance']
			pool = pd.DataFrame(np.zeros((self.maturity+1,7)), columns = columns)
			pool.loc[0,'Balance'] = balance
			for month in range(1, term+1):
				prev_balance = pool.loc[month-1,'Balance']
				pool.loc[month, 'PMT'] = self.coupon_payment(r_month, term - (month - 1), prev_balance)
				pool.loc[month, 'Interest'] = prev_balance*r_month
				pool.loc[month, 'Principal'] = prev_balance if pool.loc[month, 'PMT'] - pool.loc[month, 'Interest'] > prev_balance else pool.loc[month, 'PMT'] - pool.loc[month, 'Interest']
				pool.loc[month, 'CPR'] = 0.06*PSA*min(1, (month + age)/30)
				pool.loc[month, 'SMM'] = 1 - (1 - pool.loc[month, 'CPR'])**(1/12)
				pool.loc[month, 'Prepay CF'] = pool.loc[month, 'SMM']*(prev_balance - pool.loc[month, 'Principal'])
				pool.loc[month, 'Balance'] = prev_balance - pool.loc[month, 'Principal'] - pool.loc[month, 'Prepay CF']
			pools.append(pool)

		for pool in pools:
			self.pool_summary['Total Principal'] += pool['Principal'] + pool['Prepay CF']
			self.pool_summary['Total Interest'] += pool['Interest']
			self.pool_summary['Balance'] += pool['Balance']

		for month in range(1, self.maturity+1):
			self.pool_summary.loc[month, 'Interest Available to CMO'] = self.pool_interest_rate/12*self.pool_summary.loc[month - 1, 'Balance']

	def calculate_pool_groups_proportions(self):
		self.principal_groups_proportions = {}
		total_balance = 0
		for group in self.principal_sequential_pay:
			for cl in self.principal_sequential_pay[group]:
				self.principal_groups_proportions[group] = self.principal_groups_proportions.get(group, 0) + self.classes_info.loc[cl, 'Original Balance']
				total_balance += self.classes_info.loc[cl, 'Original Balance']
		for group in self.principal_groups_proportions:
			self.principal_groups_proportions[group] = float(self.principal_groups_proportions[group])/total_balance

	def calculate_classes_cf(self,simulated_r):
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
		self.total_cf['R'] = coupon_differential + self.total_cf.iloc[:,0:-1].sum(axis=1)



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

			################################################
			#Z_up_i = 1 # FOR NOW, REMEMBER ME TO ERASE IT AFTER
			#Z_dn_i = 1
			# Just for testing price of undiscounted cash flows
			################################################

			for cl_ind in range(len(self.classes)):
				cashflows = np.array(self.total_cf.iloc[:, cl_ind])
				simulated_prices[i, cl_ind] = (np.sum(cashflows*Z_up_i) + np.sum(cashflows*Z_dn_i))/2

		summary_np = np.zeros((2,len(self.classes)))
		summary_np[0, :] = simulated_prices.mean(0)
		summary_np[1, :] = simulated_prices.std(0)/np.sqrt(n)

		simulation_summary = pd.DataFrame(summary_np, columns=self.classes)
		simulation_summary.index = ['Average price', 'Standard error']
		print(simulation_summary)
		return simulation_summary
