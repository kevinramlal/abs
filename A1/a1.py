"""///
ABS Assignment 1
Members: Kevin, Nico, Romain, Sherry, Trilok
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import datetime

#-----------------------------------------------------------------
#Question 1: Hull-White Model
#-----------------------------------------------------------------


#----Importing Data ---#
fwds = pd.read_csv('fwds_20040830.csv',header = None, names = ['Fwds'])
zero = pd.read_csv('zero_rates_20040830.csv',header = None, names = ['Zero'])
zr = np.array(zero['Zero'])
zr_temp = np.append([None],zr) #Need to have initial value of None since dates start at 9_01_2004
master_rates = pd.DataFrame()
master_rates['Zero'] = zr_temp

atm_cap = pd.read_csv('atm_cap.csv', header = None) #extracted from 20040830_usd23_atm_caps file
atm_cap.columns = ['Maturity', 'Black Imp Vol']


#Import date column and format 
dates = pd.read_csv('dates.csv',header = None, dtype = str) #from Bloomberg files USD23 tab
dates = pd.to_datetime(dates[0])
dates.columns = ['Date']

#

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
master_rates['Dates'] =np.array(dates) #We don't need 2004-09-01 for zeros

#get 30/360 convention
master_rates['T_30_360'] = np.array(dates.apply(lambda x: 
						(x - dates[0]).days - ((x-dates[0]).days)%30))/360

master_rates['Discount'] = 1/(1+(master_rates['Zero']/100)/2)**(2*master_rates['T_30_360'])

# #Plot of Discount Factors
# plt.plot(master_rates['Dates'],master_rates['Discount'], 'bo--', markersize = '4')
# plt.xlabel('Maturity')
# plt.ylabel('Discount Factor')
# plt.title('Discount Factor vs Maturity')
# plt.show()

print("a) Discount Factors \n", master_rates[['Dates','Zero','Discount']].head(25), "\n")

#-----------------------------------------------------------------
#b) Calculate quarterly-compounded forward rates between each maturity
#-----------------------------------------------------------------

#Get ACT/360 Convention
master_rates['T_ACT_360'] = np.array(dates.apply(lambda x: 
						(x - dates[0]).days))/360 

#T_i - T_i-1 where T_i is ACT/360 convention
master_rates['Tau'] = master_rates['T_ACT_360'].diff(1)

#Forward Rates 
forwards = np.array((1/master_rates['Tau'])*\
						((-master_rates['Discount'].diff(1))/master_rates['Discount']))
master_rates['Forward'] = forwards

print("b) Forward Rates \n", master_rates[['Dates','Discount','Forward']].head(25), "\n")

#-----------------------------------------------------------------
#c) Calculate quarterly-compounded forward rates between each maturity
#-----------------------------------------------------------------

cap_master_df = pd.DataFrame()
cap_master_df['Maturity'] = atm_cap['Maturity']

strike = []

#The following code corresponds to the equation for X_n in c)
for m in cap_master_df['Maturity']:
	x_n = np.sum(master_rates['Tau'][2:4*m +1]*\
			master_rates['Forward'][2:4*m +1]*master_rates['Discount'][2:4*m +1])/\
			np.sum(master_rates['Tau'][2:4*m +1]*master_rates['Discount'][2:4*m +1])
	strike.append(x_n)


cap_master_df['ATM Strike'] = strike

print("c) ATM Strike Rates vs Maturity \n", cap_master_df, "\n")


#-----------------------------------------------------------------
#d) Calculate quarterly-compounded forward rates between each maturity
#-----------------------------------------------------------------

#Notes
#implement Black's Formula 
#For this question we access forward libor rates and flat vols
#Caps have quarterly caplets in this question
#summing over tau's from 0 to expiry (multiply Maturity by 4)
#use the same flat vol for all quarterly caplets when calculating a specific cap


flat_vols = np.array(atm_cap['Black Imp Vol'])
print(flat_vols)

forward_libor = np.array(fwds['Fwds'])
print(forward_libor,len(forward_libor)) #should be 30

cap_master_df['Flat_Vol'] = flat_vols

def d_1_2(flat_vol,forward_libor,t,strike,type = 1):
	d1 = (1/flat_vol*np.sqrt(t))*np.log(forward/strike) + 0.5*flat_vol*np.sqrt(t)
	if type == 1:
		return d1
	elif type == 2:
		return d1 - flat_vol*np.sqrt(t)

# def caplet(master_rates,t_i,)


# if __name__ == '__main__':
# 		# print(dates)
# 		# print(master_rates.head(25))
# 		print('kevin')
