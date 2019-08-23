
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
			balance = pools_info.loc[pool_index, 'Original Balance']
			r_month = pools_info.loc[pool_index, 'WAC']/12/100
			term = pools_info.loc[pool_index, 'Term']
			age = pools_info.loc[pool_index, 'Age']
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
				self.principal_groups_proportions[group] = self.principal_groups_proportions.get(group, 0) + classes_info.loc[cl, 'Original Balance']
				total_balance += classes_info.loc[cl, 'Original Balance']
		for group in self.principal_groups_proportions:
			self.principal_groups_proportions[group] = float(self.principal_groups_proportions[group])/total_balance

	def calculate_classes_cf(self):
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
		self.classes_interest_cf['R'] = self.pool_summary['Interest Available to CMO'] - self.classes_interest_cf.sum(1)
		self.total_cf = self.classes_principal + self.classes_interest_cf

		print(self.total_cf)


if __name__ == '__main__':

	# General info
	today = '8/15/2004'
	first_payment_date = '9/15/2004'
	pool_interest_rate = 0.05

	# General information of pools
	pools_info = pd.read_csv('pools_general_info.csv', thousands=',')

	# General information of classes
	classes_info = pd.read_csv('classes_general_info.csv', thousands=',')

	# Allocation sequence for principal
	principal_sequential_pay = {'1': ['CA','CY'], '2': ['CG','VE','CM','GZ','TC','CZ']}

	# Accruals accounts sequence
	accruals_sequential_pay = {'GZ': ['VE','CM'], 'CZ': ['CG','VE','CM','GZ','TC']}

	hw_remic = REMIC(today, first_payment_date, pool_interest_rate, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay)
	hw_remic.calculate_pool_cf(1)
	hw_remic.calculate_classes_cf()
	
	print(pools_info)
	print(classes_info)
	print(principal_sequential_pay)
	print(accruals_sequential_pay)

