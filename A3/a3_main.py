
# Public libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(1,'./Given_Files')
# Private libraries
import homework1
import hazard
import remic
import a3_helper

"""
Step 1: Need to estimate 4 hazards {ARM/FRM} x {Prepayment/Default}
Prepayment is the same as the coupon gap in HW2
Default is defined as LTV : (remaining balance)/(home price)
"""

#arm_data = pd.read_csv('./Given_Files/ARM_perf.csv')
#frm_data = pd.read_csv('./Given_Files/FRM_perf.csv')


tables_file = open("tables_latex_format.txt","w")
# Dynamic Hazard Model
#FRM
hz_frm_data = pd.read_csv('./Given_Files/FRM_perf.csv')
hz_frm_prepay = hazard.Hazard(hz_frm_data, prepay_col="Prepayment_indicator", end_col="Loan_age", beg_col="period_beginning", end_max=44, cov_cols=["Spread", "spring_summer"], tables_file=tables_file, show_prints=True, show_plots=False)
optimize_dynamic_theta = True # True to run next optimization and False to use precalculated values.
hz_frm_prepay.param_estimate_dynamic(optimize_flag=optimize_dynamic_theta, theta=[1.196813, 0.013147, -0.048916, -0.215195])
#hw_remic.simulation_result(hz_dynamic, simulated_lagged_10_year_rates_A, 'E','F', 'Dyanamic Data')

hz_frm_default = hazard.Hazard(hz_frm_data, prepay_col="Default_indicator", end_col="Loan_age", beg_col="period_beginning", end_max=44, cov_cols=["LTV"], tables_file=tables_file, show_prints=True, show_plots=False)
optimize_dynamic_theta = True # True to run next optimization and False to use precalculated values.
hz_frm_default.param_estimate_dynamic(phist = [0.2,0.5,1],bounds = ((0.00001,np.inf),(0.00001,np.inf),(-np.inf,np.inf)),optimize_flag=optimize_dynamic_theta, theta=[ 1.556545, 0.006759, 1.402650])


#ARM
hz_arm_data = pd.read_csv('./Given_Files/ARM_perf.csv')
hz_arm_prepay = hazard.Hazard(hz_arm_data, prepay_col="Prepayment_indicator", end_col="Loan_age", beg_col="period_beginning", end_max=44, cov_cols=["Spread", "spring_summer"], tables_file=tables_file, show_prints=True, show_plots=False)
optimize_dynamic_theta = True # True to run next optimization and False to use precalculated values.
hz_arm_prepay.param_estimate_dynamic(optimize_flag=optimize_dynamic_theta, theta=[1.847123, 0.040327, -0.126106, -0.034690])
#hw_remic.simulation_result(hz_dynamic, simulated_lagged_10_year_rates_A, 'E','F', 'Dyanamic Data')

hz_arm_default = hazard.Hazard(hz_arm_data, prepay_col="Default_indicator", end_col="Loan_age", beg_col="period_beginning", end_max=44, cov_cols=["LTV"], tables_file=tables_file, show_prints=True, show_plots=False)
optimize_dynamic_theta = True # True to run next optimization and False to use precalculated values.
hz_arm_default.param_estimate_dynamic(phist = [0.2,0.5,1],bounds = ((0.00001,np.inf),(0.00001,np.inf),(-np.inf,np.inf)),optimize_flag=optimize_dynamic_theta, theta=[1.758489, 0.017101, 0.838948])

#To use our class function for calculating hazards - we need to get our data to be similiar to that of HW2
#This innvolves consolidating data per episode


#hlp = a3_helper.h3_helper()
#test = hlp.data_wrangler(arm_data)
#print(test.head()) #NEED TO GET COUPON GAP? AND DTI?

# arm_prepay = hazard.Hazard(hz_static_data, prepay_col="Prepayment_indicator", end_col="period_end",\
# beg_col="", end_max=60, cov_cols=["Spread", "spring_summer"], tables_file=tables_file, show_prints=True, show_plo

tables_file.close()
