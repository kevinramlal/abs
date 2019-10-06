
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
		 WAC: weighted average INTEREST per bond 
		 ridership_forecast:  Ridership Forecasts as np.array
		 rev_forecasts: Revenue Forecasts as np.array
		 rev_percentage: % of Revenue that goes toward paying tranches
		 simulated_rates: Simulated Rates Monte Carlo
		 maturity: Number of months until maturty

		 TODO: Is LTV needed? I assume it's 100%

	'''
	def __init__(self,tranche_list, tranche_principal, WAC, ridership_forecast, rev_forecasts,rev_percentage, simulated_rates, maturity, tables_file, show_prints=False, show_plots=False):
		self.tranche_list = tranche_list
		self.tranche_principal = tranche_principal
		self.WAC = WAC
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
		if 'R' in self.regular_classes:
			self.regular_classes.remove('R')

	
	def calculate_cashflows(self):
		"""
			Given trance information, and revenue, calculate cashflows per trance
		"""
		r_month = self.WAC/12/100
		self.balance = np.zeros((self.N, self.T)) #do we need this?
		self.balance[:,0] = self. 
		amortization_pmt = np.zeros((self.N, self.T))
		self.interest_pmt = np.zeros((self.N, self.T))
		self.principal_pmt = np.zeros((self.N, self.T))

		#TODO prepayment
		self.prepayment = np.zeros((self.N, self.T))

		#TODO default??
		self.default = np.zeros((self.N, self.T))

		#TODO default managment

		#Tranches
		self.bonds_balance = {k:np.zeros((self.N,self.T)) for k in tranche_list} #initialize our bonds principal
		self.bonds_interest = {k:np.zeros((self.N,self.T)) for k in tranche_list} #keep track of interest
		# self.bonds_principal = {k:np.zeros((self.N,self.T)) for k in tranche_list}
		# self.bonds_extra_principal = {k:np.zeros((self.N,self.T)) for k in tranche_list} # Distributed due to excess spread after overcollateralization target has been achieved.
		# self.bonds_principal_default = {k:np.zeros((self.N,self.T)) for k in tranche_list}
		for i in range(len(tranche_list)):
			self.bonds_balance[tranche_list[i]][:,0] = self.tranche_principal[i] #need to make sure balance is in same order as tranches

		#OK! Time for cash flows
		for month in range(1,self.T):
			pmt = self.revenue[month]*self.rev_percentage #monthly revenue x % 

			#First run through
			for cl in self.tranche_list:
				prev_balance = self.bonds_balance[cl][:,month-1]
				amortized_pmt = self.coupon_payment(r_month, self.T - month , prev_balance) #scheduled principal payments
				interest_accrued = prev_balance*r_month
				
				#First we pay interest
				if pmt - interest_pmt > 0:
					pmt -= interest_accrued
					interest_accrued = 0

				else:
					interest_accrued -= pmt
					pmt = 0

				#Next we pay Scheduled principal if we have money 
				if pmt - amortized_pmt > 0:
					pmt -= amortized_pmt
					new_bal = prev_balance - amortized_pmt

				else:
					new_bal -= pmt

				self.bonds_balance[cl][:,month] = new_bal + interest_accrued
				self.bonds_interest[cl][:,month] = interest_accrued

			#Second run through if we have extra money
			i = 0
			while pmt > 0:
				pmt_temp = max(0,pmt - self.bonds_balance[self.tranche_list[i][:month]]) #Remaining pmt is either 0 or left over from tranches in priority order
				self.bonds_balance[self.tranche_list[i][:month]] = max(0,self.bonds_balance[self.tranche_list[i][:month]] - pmt) #Update balance
				pmt = pmt_temp
				i += 1
				













