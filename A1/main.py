
# Public libraries
import pandas as pd

# Private libraries
import homework1
import hazard
import remic

#
# Through out the code we use _A to indicate the object has both normal paths and antithetic paths
#

# Term struture model
hw1 = homework1.Homework1(show_prints=False, show_plots=False)
hw1.fit_term_structure_model()
simulated_rates_A, simulated_Z_A = hw1.simulate_interest_rates(n=10)

# Fit Hazard Model
hz_static_data = pd.read_csv('static.csv', thousands=',')
hz = hazard.Hazard(hz_static_data, prepay_col="prepay", end_col="period_end", beg_col="", end_max=60, cov_cols=["cpn_gap", "summer"], show_prints=True, show_plots=False)
hz.fit_parameters_grad()
hz.parameters_se()

# REMIC cashflows
start_date = '8/15/2004'
first_payment_date = '9/15/2004'
pool_interest_rate = 0.05
pools_info = pd.read_csv('pools_general_info.csv', thousands=',')
classes_info = pd.read_csv('classes_general_info.csv', thousands=',')
principal_sequential_pay = {'1': ['CA','CY'], '2': ['CG','VE','CM','GZ','TC','CZ']}
accruals_sequential_pay = {'GZ': ['VE','CM'], 'CZ': ['CG','VE','CM','GZ','TC']}
simulated_lagged_10_year_rates_A = hw1.calculate_T_year_rate_APR(simulated_rates_A, lag=3, horizon=10)
hw_remic = remic.REMIC(start_date, first_payment_date, pool_interest_rate, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay, simulated_rates_A, simulated_Z_A, show_prints=True, show_plots=False)
hw_remic.simulation_result(hz, simulated_lagged_10_year_rates_A)

# REMIC pricing
#hw_remic.price_classes(simulated_Z_A)
#hw_remic.calculate_durations_and_convexities(dr=0.0001, dt=1/12)


#Calculating Ooption Adjusted Spreads
#hw_remic.find_oas_classes(simulated_rates_A)
