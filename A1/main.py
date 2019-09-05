
# Public libraries
import pandas as pd

# Private libraries
import homework1
import hazard
import remic

# Term struture model
hw1 = homework1.Homework1(show_prints=False, show_plots=False)
hw1.fit_term_structure_model()
simulated_rates, simulated_Z = hw1.simulate_interest_rates(n=100)

# Fit Hazard Model
hz_static_data = pd.read_csv('static.csv', thousands=',')
hz = hazard.Hazard(hz_static_data, prepay_col="prepay", end_col="period_end", end_max=60, cov_cols=["cpn_gap", "summer"], show_prints=True, show_plots=True)
hz.fit_parameters_grad()
hz.parameters_hessian()
hz.parameters_se()

# REMIC cashflows
today = '8/15/2004'
first_payment_date = '9/15/2004'
pool_interest_rate = 0.05
pools_info = pd.read_csv('pools_general_info.csv', thousands=',')
classes_info = pd.read_csv('classes_general_info.csv', thousands=',')
principal_sequential_pay = {'1': ['CA','CY'], '2': ['CG','VE','CM','GZ','TC','CZ']}
accruals_sequential_pay = {'GZ': ['VE','CM'], 'CZ': ['CG','VE','CM','GZ','TC']}
hw_remic = remic.REMIC(today, first_payment_date, pool_interest_rate, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay, show_prints=False, show_plots=False)
hw_remic.calculate_pool_cf(PSA=1.5)
hw_remic.calculate_classes_cf()

# REMIC pricing
hw_remic.price_classes(simulated_Z)
hw_remic.calculate_durations_and_convexities(dr=0.0001, dt=1/12)


#Calculating Ooption Adjusted Spreads
#hw_remic.find_oas_classes(simulated_rates)
