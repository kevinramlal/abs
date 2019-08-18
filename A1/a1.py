"""///
ABS Assignment 1
Members: Kevin, Nico, Romain, Sherry, Trilok
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import datetime

#----Part 1: Importing Data and Calibrating Data-----#
fwds = pd.read_csv('fwds_20040830.csv',header = None, names = ['Fwds'])
zero_rates = pd.read_csv('zero_rates_20040830.csv',header = None, names = ['Zero'])
atm_cap = pd.read_csv('atm_cap.csv', header = None) #extracted from 20040830_usd23_atm_caps file
atm_cap.columns = ['Maturity', 'Black Imp Vol']

#Import date column and format 
dates = pd.read_csv('dates.csv',header = None, dtype = str) #from Bloomberg files USD23 tab
dates = pd.to_datetime(dates[0])
dates.columns = ['Date']


# ------(Optional) Plot of Black Implied Vols for ATM Caps ------
# plt.plot(atm_cap['Maturity'], atm_cap['Black Imp Vol'], 'bo--')
# plt.title('Black Implied Flat Vol for ATM Caps')
# plt.xlabel('Maturities')
# plt.ylabel('Black Imp Vol (%)')
# plt.show()


#-----------------------------------------------------------------
#a) Caclulate Discount factors Z(D_i)

#-----------------------------------------------------------------

#Combine dates and zero rates 
start_date = dates[0]
zero_rates['Dates'] =np.array(dates[1:]) #We don't need 2004-09-01 for zeros

#get 30/360 convention
zero_rates['T_30_360'] = np.array(dates[1:].apply(lambda x: 
						(x - dates[0]).days - ((x-dates[0]).days)%30))/360

zero_rates['Discount'] = 1/(1+zero_rates['Zero']/2)**(2*zero_rates['T_30_360'])

plt.plot(zero_rates['Dates'],zero_rates['Discount'], 'bo--', markersize = '4')
plt.xlabel('Maturity')
plt.ylabel('Discount Factor')
plt.title('Discount Factor vs Maturity')
plt.show()


print(zero_rates.head(25))


if __name__ == '__main__':
		# print(dates)
		print(zero_rates.head(25))