#a3_helper_functions

import pandas as pd
import numpy as np

class h3_helper:
	def __init__(self):
		return None

	def data_wrangler(self,df):
		"df - either ARM or FRM data file"
		consolidated_df = pd.DataFrame()
		consolidated_df['id_loan'] = list(set(df['Loan_id'])) #individual loans
		consolidated_df['period_end'] = np.array(df[['Loan_id','Loan_age']].groupby('Loan_id').max())
		consolidated_df['period_begin'] = np.array(df[['Loan_id','period_beginning']].groupby('Loan_id').min())
		consolidated_df['orig_ltv'] = np.array(df[['Loan_id','LTV']].groupby('Loan_id').min())
		consolidated_df['orig_upb'] = np.array(df[['Loan_id','Remaining_balance']].groupby('Loan_id').min())	
		consolidated_df['Act_endg_upb'] = np.array(df[['Loan_id','Remaining_balance']].groupby('Loan_id').max())	
		consolidated_df['summer'] = np.array(df[['Loan_id','spring_summer']].groupby('Loan_id').max())

		return consolidated_df	