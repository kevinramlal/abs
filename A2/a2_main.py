
# Public libraries
import pandas as pd
import os 
import sys 

sys.path.insert(1,'./Given_Files')
# Private libraries
import homework1
import hazard
import remic

#
# Through out the code we use _A to indicate the object has both normal paths and antithetic paths
#

tables_file = open("Results/tables_latex_format.txt","w")

# Term struture model
n_simulations = 20 # Has to be even
hw1 = homework1.Homework1(show_prints=False, show_plots=False)
hw1.fit_term_structure_model()
simulated_rates_A = hw1.simulate_interest_rates(n=n_simulations)

# REMIC initialization
start_date = '8/15/2004'
first_payment_date = '9/15/2004'
pool_interest_rate = 0.05
pools_info = pd.read_csv('./Given_Files/pools_general_info.csv', thousands=',')
classes_info = pd.read_csv('./Given_Files/classes_general_info.csv', thousands=',')
principal_sequential_pay = {'1': ['CA','CY'], '2': ['CG','VE','CM','GZ','TC','CZ']}
accruals_sequential_pay = {'GZ': ['VE','CM'], 'CZ': ['CG','VE','CM','GZ','TC']}
previous_rates = [0.025313, 0.025587, 0.02344]
simulated_lagged_10_year_rates_A = hw1.calculate_T_year_rate_APR(simulated_rates_A, lag=3, horizon=10, previous_rates=previous_rates)
hw_remic = remic.REMIC(start_date, first_payment_date, pool_interest_rate, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay, simulated_rates_A, tables_file, show_prints=True, show_plots=False)

# Static Hazard Model and consequent REMIC bonds results
hz_static_data = pd.read_csv('./Given_Files/static.csv', thousands=',')
hz = hazard.Hazard(hz_static_data, prepay_col="prepay", end_col="period_end", beg_col="", end_max=60, cov_cols=["cpn_gap", "summer"], tables_file=tables_file, show_prints=True, show_plots=False)
hz.fit_parameters_grad()
hz.parameters_se()
hw_remic.simulation_result(hz, simulated_lagged_10_year_rates_A, 'B','C', 'Static Data')

# Dynamic Hazard Model and consequent REMIC bonds results
hz_dynamic_data = pd.read_csv('./Given_Files/dynamic.csv', thousands=',')
hz_dynamic = hazard.Hazard(hz_dynamic_data, prepay_col="prepay", end_col="period_end", beg_col="period_begin", end_max=60, cov_cols=["cpn_gap", "summer"], tables_file=tables_file, show_prints=True, show_plots=False)
optimize_dynamic_theta = False # True to run next optimization and False to use precalculated values. Optimization takes around 15 minutes in regular CPU.
hz_dynamic.param_estimate_dynamic(optimize_flag=optimize_dynamic_theta, theta=[1.451547, 0.009143, 0.591170, 0.069740])
hw_remic.simulation_result(hz_dynamic, simulated_lagged_10_year_rates_A, 'E','F', 'Dyanamic Data')

tables_file.close()
