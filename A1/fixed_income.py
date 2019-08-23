
import pandas as pd
import numpy as np

class FixedIncome:
	'''
		This class contains all functions and algorithms needed to estimate fixed income models
	'''

	def __init__(self):
		pass


	# ----------------------- #
	#  Day count conventions  #
	# ----------------------- #

	def get_days_act_360(self, start_date, end_date):
	    return (end_date-start_date).days/360
	    
	def get_days_30I_360(self, start_date, end_date):
	    Y1 = start_date.year
	    Y2 = end_date.year
	    M1 = start_date.month
	    M2 = end_date.month
	    D1 = start_date.day
	    D2 = end_date.day
	    if D1==31:
	        D1=30
	    if D2==31:
	        D2=30
	    return (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1))/360