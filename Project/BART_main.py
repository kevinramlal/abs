
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
#    tranche_principal = [317774500, 316524000,312945800,310326300,308429400]
    bond_spread = [0.0005, 0.0015, 0.0035, 0.0055, 0.0075]
    base_coupon_rate = 0.026414139  #0.02648
rev_percentage = 0.60 # 40% of ridership revenue is used to fund the entirety of non-labour operating expenses.
maturity = (33-18)*12 # months
bart = bart_class.BART(tranche_list, tranche_principal, bond_spread, base_coupon_rate, rev_percentage, simulated_rates_A, maturity, tables_file, show_prints=True, show_plots=False)
bart.forecast_revenue()
bart.calculate_cashflows()
bart.calculate_bond_prices(show_prints=True)
bart.calculate_duration_convexity()

def optimized_rev(base_coupon):
	bc = base_coupon
	print(bc)
	bart = bart_class.BART(tranche_list, tranche_principal, bond_spread, bc, rev_percentage, simulated_rates_A, maturity, tables_file, show_prints=True, show_plots=False)
	bart.forecast_revenue()
	bart.calculate_cashflows()
	residual = bart.calculate_bond_prices()[0]
	return residual
# 	return residual

#plotting balance
balances = bart.bonds_balance
balances_avg = [np.mean(v,axis = 0) for k,v in balances.items()]
x = np.arange(1,bart.T+1,1)

for i in range(len(bart.regular_classes)):
	plt.plot(x,balances_avg[i],label = bart.regular_classes[i])
	# plt.legend()
	# plt.show()
plt.xlabel("Month")
plt.ylabel("Balance ($100MM)")
plt.title("Bond Balances")
plt.legend()
plt.show()





tables_file.close()
