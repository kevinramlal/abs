
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
	def __init__(self,tranche_list, tranche_principal, bond_spread, ridership_forecast, rev_forecasts,rev_percentage, simulated_rates, maturity, tables_file, show_prints=False, show_plots=False):
		self.tranche_list = tranche_list
		self.tranche_principal = tranche_principal
		self.bond_spread = spread
		self.ridership_forecast = ridership_forecast
		self.rev_forecasts = rev_forecasts
		self.rev_percentage = rev_percentage
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

	
	def calculate_cashflows(self):
		"""
			Given tranche information, and revenue, calculate cashflows per trance
		"""
		bond_spread_dict ={self.regular_classes[i]:self.bond_spread[i] for i in range(len(self.regular_classes))}
		self.residual = np.zeros((self.N,self.T))

		#Tranches
		self.bonds_balance = {k:np.zeros((self.N,self.T)) for k in self.regular_classes} #initialize our bonds principal
		self.bonds_interest = {k:np.zeros((self.N,self.T)) for k in self.regular_classes} #keep track of interest
		for i in range(len(self.regular_classes)):
			self.bonds_balance[self.regular_classes[i]][:,0] = self.tranche_principal[i] #This intializes all simulations 

		#OK! Time for cash flows
		for month in range(1,self.T):
			pmt = self.revenue[:,month]*self.rev_percentage #monthly revenue x % for ALL simulations

			#First run through
			for cl in self.regular_classes:
				prev_balance = self.bonds_balance[cl][:,month-1] #all simulations array for month 
				r_month = self.simulated_rates[:,month-1] + bond_spread_dict[cl] #Add spread to each simulated libor rate at that month 
				amortized_pmt = self.coupon_payment(r_month, self.T - month , prev_balance) #This should work as numpy array component wise multiplication
				interest_accrued = prev_balance*r_month #also numpy array multiplication 
				
				pmt, self.bonds_balance[cl][:,month], self.bonds_interest[cl][:,month] = self.first_pass(pmt,prev_balance,amortized_pmt,interest_accrued)

			pmt = self.second_pass(pmt,self.bonds_principal,month) #if any extra pmt then start paying off principal from tranches 

			residual[:,month] = pmt #should be all zeroes unless theres left over cash after paying off ALL tranches 

		return self.bonds_balance, self.bonds_interest, self.residual

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

		water_fall_payer = lambda x: max(0,x) #x should be Interest - Payment for interest, then Amort Bal - Payment for principal
		pmt_after_pay = lambda x: max(0,x) #reverse order of operations as previous 

		interest_accrued = np.array(list(map(water_fall_payer,(interest_accrued_array - pmt_array))))
		pmt_after_interest = np.array(list(map(pmt_after_pay,(pmt_array - interest_accrued_array))))

		amort_pay_deduct =  np.array(list(map(water_fall_payer,(amort_pmt_array - pmt_after_interest)))) #Either 0 
		pmt_after_princpal = np.array(list(map(pmt_after_pay, (pmt_after_interest - amort_pmt_array))))
		new_balance  = prev_balance_array - amort_pmt_array + amort_pay_deduct + interest_accrued #(Prev Balance - Amortization + Deduction from Amort (either 0 or amort - pmt)) + Interest Accrus 

		pmt_new = pmt_after_princpal

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

		return pmt_new,new_balance,interest_accrued

	def extra_pass(self,pmt_array,bonds_balance,month):
		"""
		TODO: Vectorize 
		"""

		pmt_new = []
		 
		for j in range(len(pmt_array)):
			pmt_sim = pmt_array[j]

			i = 0
			while pmt_sim > 0:
				pmt_temp = max(0,pmt_sim - self.bonds_balance[self.regular_classes[i]][j,month]) #Remaining pmt is either 0 or left over from tranches in priority order
				self.bonds_balance[self.regular_classes[i]][j,month] = max(0,self.bonds_balance[self.regular_classes[i]][j,month] - pmt_sim) #Update balance
				pmt_sim = pmt_temp
				i += 1
			pmt_new.append(pmt_sim) #should all be zeroes theoretically unless extra left over afte ALL tranche principals paid off 

		return pmt_new


	
	def coupon_payment(self, r_month, months_remaining, balance):
		return r_month*balance/(1-1/(1+r_month)**months_remaining)




#TODO: Price by discounting 
#TODO: Duration, Convexity, OAS - to make it par
#TODO: CDS Pricing
#TODO: 





