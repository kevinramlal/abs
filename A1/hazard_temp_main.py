
import pandas as pd
import numpy as np
import hazard

pd.set_option('display.max_columns', 13)

hz_static_data = pd.read_csv('static.csv', thousands=',')
print(hz_static_data)

hz = hazard.Hazard(hz_static_data, prepay_col="prepay", end_col="period_end", end_max=60, cov_cols=["cpn_gap", "summer"])
#hz.fit_parameters_brute()
hz.fit_parameters_grad()
hz.baseline_hazard()



