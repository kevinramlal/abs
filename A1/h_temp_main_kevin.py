
import pandas as pd
import numpy as np
import hazard_kevin

pd.set_option('display.max_columns', 13)

hz_static_data = pd.read_csv('static.csv', thousands=',')

hz = hazard_kevin.Hazard(hz_static_data, prepay_col="prepay", end_col="period_end",beg_col="period_begin", end_max=60, cov_cols=["cpn_gap", "summer"])
#hz.fit_parameters_brute()
hz.fit_parameters_grad()
#hz.baseline_hazard()
hz.parameters_hessian()
hz.parameters_se()

#Part d
##Dynamic covariates
hz_dynamic_data = pd.read_csv('dynamic.csv', thousands=',')
hz_dynamic = hazard_kevin.Hazard(hz_dynamic_data, prepay_col="prepay", end_col="period_end",beg_col="period_begin", end_max=60, cov_cols=["cpn_gap", "summer"])
print(hz_dynamic.param_estimate_dynamic())
