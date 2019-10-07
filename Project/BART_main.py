
# Public libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import minimize 
sys.path.insert(1,'./Given_Files')

# Private libraries
import a3_helper as a3
import hazard
import remic
import bart_class

tables_file = open("tables_latex_format.txt","w")

# Term struture model
n_simulations = 1000 # Has to be even
hw1 = a3.hull_white_fit()
simulated_rates_A = hw1.simulate_interest_rates(n=n_simulations)



# Bart initialization
tranche_list = ['A1','A2','M1','M2','M3','R']

tranching_system = 2

if (tranching_system == 1):
    principal_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    tranche_principal = [x * 1566000000 for x in principal_proportions]
    print (tranche_principal)
    bond_spread = [0.0008, 0.0018, 0.0038, 0.0055, 0.0115]
    base_coupon_rate = 0.04
elif (tranching_system == 2):
    tranche_principal = [313200000, 313200000, 313200000, 313200000, 313200000]
    bond_spread = [0.0005, 0.0015, 0.0035, 0.0055, 0.0075]
    base_coupon_rate = 0.025

rev_percentage = 0.60 # 40% of ridership revenue is used to fund the entirety of non-labour operating expenses.
maturity = (33-18)*12 # months
bart = bart_class.BART(tranche_list, tranche_principal, bond_spread, base_coupon_rate, rev_percentage, simulated_rates_A, maturity, tables_file, show_prints=True, show_plots=False)
bart.forecast_revenue()
bart.calculate_cashflows()
bart.calculate_bond_prices()

def optimized_rev(base_coupon):
	bc = base_coupon
	print(bc)
	bart = bart_class.BART(tranche_list, tranche_principal, bond_spread, bc, rev_percentage, simulated_rates_A, maturity, tables_file, show_prints=True, show_plots=False)
	bart.forecast_revenue()
	bart.calculate_cashflows()
	residual = bart.calculate_bond_prices()
	return residual
# 	return residual

# cons = ({'type': 'ineq', 'fun': lambda x:  (1-x[0])},
# 	{'type': 'ineq', 'fun': lambda x:  x[0]})
# res = minimize(optimized_rev, x0 = 0.04, method='Nelder-Mead', constraints = cons)
# print("KEVINS MAGIC: ", res.x)

# def kevin_optimizer(start_percentage):
# 	residual = 1000000
# 	rev_p = start_percentage
# 	while abs(residual) > 100000:
# 		print(rev_p)
# 		bart = bart_class.BART(tranche_list, tranche_principal, bond_spread, base_coupon_rate, rev_p, simulated_rates_A, maturity, tables_file, show_prints=True, show_plots=False)
# 		bart.forecast_revenue()
# 		bart.calculate_cashflows()
# 		residual = bart.calculate_bond_prices()
# 		if np.sign(residual) == 1:
# 			rev_p += 0.01
# 		else:
# 			rev_p -= 0.01
# 	return rev_p

# print(kevin_optimizer(0.4))

#plotting balance
balances = bart.bonds_balance
balances_avg = [np.mean(v,axis = 0) for k,v in balances.items()]
x = np.arange(1,bart.T+1,1)

for i in range(len(bart.regular_classes)):
	plt.plot(x,balances_avg[i],label = bart.regular_classes[i])
	# plt.legend()
	# plt.show()
plt.legend()
plt.show()


## Pools initialization
#pool_origination_date = '3/27/2006'
#evaluation_date = '6/30/2009'
#pools_info = pd.read_csv('./Given_Files/pools_general_info.csv', thousands=',')
#classes_info = pd.read_csv('./Given_Files/classes_general_info.csv', thousands=',')
#principal_sequential_pay = {'1': ['A2','A3','M1','M2','M3','M4','M5','M6','M7','M8']}
#accruals_sequential_pay = {}
## previous_rates = [0.026647, 0.031206, 0.034613] # TREASURY 10 YR
#previous_rates = [0.012075, 0.011804872, 0.01149797086] # LIBOR 3M
#simulated_lagged_10_year_rates_A = hw1.calculate_T_year_rate_APR(simulated_rates_A, lag=3, horizon=10, previous_rates=previous_rates)
#hw_remic = remic.REMIC(pool_origination_date, evaluation_date, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay, simulated_rates_A, tables_file, show_prints=True, show_plots=True)
#
## House price evolutions for FRM and ARM
#rental_flow_rate = 0.025
#vol_house_prices = 0.12
#hw_remic.simulate_house_prices(n_simulations, rental_flow_rate, vol_house_prices)
#
## Hazard models
#optimize_hazard_models = True # True to run all hazard models optimizations.
#
## FRM prepayment
#hz_frm_data = pd.read_csv('./Given_Files/FRM_perf.csv')
#hz_frm_prepay = hazard.Hazard(hz_frm_data, prepay_col="Prepayment_indicator", end_col="Loan_age", beg_col="period_beginning", end_max=44, cov_cols=["Spread", "spring_summer"], tables_file=tables_file, show_prints=True, show_plots=False)
#hz_frm_prepay.param_estimate_dynamic(optimize_flag=optimize_hazard_models, theta=[1.196813, 0.013147, -0.048916, -0.215195])
#
## FRM default
#hz_frm_default = hazard.Hazard(hz_frm_data, prepay_col="Default_indicator", end_col="Loan_age", beg_col="period_beginning", end_max=44, cov_cols=["LTV"], tables_file=tables_file, show_prints=True, show_plots=False)
#hz_frm_default.param_estimate_dynamic(phist = [0.2,0.5,1],bounds = ((0.00001,np.inf),(0.00001,np.inf),(-np.inf,np.inf)),optimize_flag=optimize_hazard_models, theta=[ 1.556545, 0.006759, 1.402650])
#
## ARM prepayment
#hz_arm_data = pd.read_csv('./Given_Files/ARM_perf.csv')
#hz_arm_prepay = hazard.Hazard(hz_arm_data, prepay_col="Prepayment_indicator", end_col="Loan_age", beg_col="period_beginning", end_max=44, cov_cols=["Spread", "spring_summer"], tables_file=tables_file, show_prints=True, show_plots=False)
#hz_arm_prepay.param_estimate_dynamic(optimize_flag=optimize_hazard_models, theta=[1.847123, 0.040327, -0.126106, -0.034690])
#
## ARM default
#hz_arm_default = hazard.Hazard(hz_arm_data, prepay_col="Default_indicator", end_col="Loan_age", beg_col="period_beginning", end_max=44, cov_cols=["LTV"], tables_file=tables_file, show_prints=True, show_plots=False)
#hz_arm_default.param_estimate_dynamic(phist = [0.2,0.5,1],bounds = ((0.00001,np.inf),(0.00001,np.inf),(-np.inf,np.inf)),optimize_flag=optimize_hazard_models, theta=[1.758489, 0.017101, 0.838948])
#
## Bond pricing
#hw_remic.simulation_result(hz_frm_prepay, hz_frm_default, hz_arm_prepay, hz_arm_default, simulated_lagged_10_year_rates_A)



tables_file.close()
