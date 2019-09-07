"""
ABS Assignment 1
Members: Kevin, Nico, Romain, Sherry, Sagnik

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm
from scipy.optimize import minimize
from scipy import optimize
from scipy import interpolate
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Custom made classes
import fixed_income
from utilities import *
import os
# print(os.path.dirname(__file__))

class Homework1:

    

    def __init__(self, show_prints=False, show_plots=False):
        self.show_prints = show_prints
        self.show_plots = show_plots
        self.fi = fixed_income.FixedIncome()

    def fit_term_structure_model(self):
        home_dir = os.path.dirname(__file__)

        #-----------------------------------------------------------------
        # Importing Data and Calibrating Data
        #-----------------------------------------------------------------

        fwds = pd.read_csv(home_dir + "/fwds_20040830.csv",header = None, names = ['Fwds'])
        zero = pd.read_csv(home_dir +'/zero_rates_custom.csv',header = None, names = ['Zero'])

        zr = np.array(zero['Zero'])
        zr_temp = np.append([None],zr) #Need to have initial value of None since dates start at 9_01_2004
        self.master_rates = pd.DataFrame()
        self.master_rates['Zero'] = zr_temp

        atm_cap = pd.read_csv(home_dir +'/atm_cap.csv', header = None) #extracted from 20040830_usd23_atm_caps file
        atm_cap.columns = ['Maturity', 'Black Imp Vol']

        #Import Expiry, Settlement Date column and format as datetime
        dates_all = pd.read_csv(home_dir +'/dates_expiry_settle.csv',dtype = str) #from Bloomberg files USD23 tab
        dates = pd.to_datetime(dates_all['Settlement Date']) #settlement dates
        dates.columns = ['Date']

        dates_settle = pd.to_datetime(dates_all['Expiry Date'])
        dates_settle.columns = ['Settle_Date']


        #-----------------------------------------------------------------
        # Caclulate Discount factors Z(D_i)
        #-----------------------------------------------------------------

        #Combine dates and zero rates
        start_date = dates[0]
        self.master_rates['Dates'] =np.array(dates) #We don't need 2004-09-01 for zeros
        self.master_rates['T_30_360'] = np.array(dates.apply(lambda x: self.fi.get_days_30I_360(start_date,x)))
        self.master_rates['Discount'] = 1/(1+(self.master_rates['Zero']/100)/2)**(2*self.master_rates['T_30_360'])
        self.master_rates['Expiry Dates'] = np.array(dates_settle)
        self.master_rates['Expiry_day_count'] = np.array(dates_settle.apply(lambda x: (x - self.master_rates['Dates'][0]).days))

        if self.show_prints:
            print("\n1a) Discount Factors \n", self.master_rates[['Dates','Zero','Discount']].head(25), "\n")
            #print(latex_table(self.master_rates[['Dates','Zero','Discount']], caption="Discount Factors", label="p1a_discount", index=False))

        if self.show_plots:
            plt.plot(self.master_rates['Dates'], self.master_rates['Discount'], 'b', ms = 6)
            plt.xlabel('Date')
            plt.ylabel('Discount Rate')
            #plt.title('Discount (Quarterly)')
            plt.savefig('1a_discount.eps', format='eps')
            plt.show()

        #-----------------------------------------------------------------
        # Calculate quarterly-compounded forward rates between each maturity
        #-----------------------------------------------------------------

        #Get ACT/360 Convention
        self.master_rates['T_ACT_360'] = np.array(dates.apply(lambda x: self.fi.get_days_act_360(start_date,x)))

        #T_i - T_i-1 where T_i is ACT/360 convention
        self.master_rates['Tau'] = self.master_rates['T_ACT_360'].diff(1)

        #Forward Rates
        forwards = np.array((1/self.master_rates['Tau'])*\
                                ((-self.master_rates['Discount'].diff(1))/self.master_rates['Discount']))

        self.master_rates['Forward'] = forwards

        if self.show_prints:
            print("\n1b) Forward Rates \n", self.master_rates[['Dates','Discount','Forward']].head(25), "\n")
            ##print(latex_table(self.master_rates[['Dates','Zero','Discount','Forward']], caption="Discount factors and forward rates", label="p1ab", index=False))

        if self.show_plots:
            plt.plot(self.master_rates['Dates'], self.master_rates['Forward'], 'b', ms = 6, label = "Forward Rate")
            plt.xlabel('Date')
            plt.ylabel('Forward Rate')
            #plt.title('Forward Rate')
            plt.savefig('1b_forward.eps', format='eps')
            plt.show()

        #-----------------------------------------------------------------
        # Calculating the at-the-money (ATM) strike rates for each of the 15 caps
        #-----------------------------------------------------------------

        cap_master_df = pd.DataFrame()
        cap_master_df['Maturity'] = atm_cap['Maturity']

        strike = []

        #The following code corresponds to the equation for X_n in c)
        for m in cap_master_df['Maturity']:
            x_n = np.sum(self.master_rates['Tau'][2:4*m +1]*\
                    self.master_rates['Forward'][2:4*m +1]*self.master_rates['Discount'][2:4*m +1])/\
                    np.sum(self.master_rates['Tau'][2:4*m +1]*self.master_rates['Discount'][2:4*m +1])
            strike.append(x_n)


        cap_master_df['ATM Strike'] = strike

        if self.show_prints:
            print("\n1c) ATM Strike Rates vs Maturity \n", cap_master_df, "\n")
            #print(latex_table(cap_master_df, caption="ATM strikes for given caps", label="p1c", index=False))

        #-----------------------------------------------------------------
        # Estimating k and sigma
        #-----------------------------------------------------------------

        #---------Implementing Blacks Formula---------------------#
        flat_vols = np.array(atm_cap['Black Imp Vol'])
        forward_libor = np.array(fwds['Fwds'])
        fwd_libor_temp = [None]
        forward_libor = np.append(fwd_libor_temp,forward_libor)
        self.master_rates['Forward_Libor'] = forward_libor #Has one extra entry at end that we cannot compute?
        cap_master_df['Flat_Vol'] = flat_vols

        def d_1_2(flat_vol,forward_libor,t,strike,type = 1):
            d1 = (1/(flat_vol*np.sqrt(t)))*np.log(forward_libor/strike) + 0.5*flat_vol*np.sqrt(t)
            if type == 1:
                return d1
            elif type == 2:
                return d1 - flat_vol*np.sqrt(t)

        def caplet_black(time_index,N, flat_vol, strike):
            """
            time_index - index based on T_30_360 convenion (example - 0.75 -> 3)
            Returns caplet value for a given time based on Black formula.
            """
            t_i = self.master_rates['Expiry_day_count'][time_index]/365
            tau = self.master_rates['Tau'][time_index +1]
            fwd = self.master_rates['Forward_Libor'][time_index+1]/100 #Note used Libor Forward
            fv = flat_vol/100
            discount = self.master_rates['Discount'][time_index+1]
            d1 = d_1_2(fv,fwd,t_i,strike)
            d2 = d_1_2(fv,fwd,t_i,strike,type =2)
            phi_d1 = norm.cdf(d1)
            phi_d2 = norm.cdf(d2)
            pv = ((N*tau)*discount*(fwd*phi_d1 - strike*phi_d2))

            return pv

        def caplet_hull_white(index,N,vol,kappa, strike_input):
            """
            Similiar parameters as Black model but needs constant vol and kappa
            """
            tau = self.master_rates ['Tau'][index+1]
            t_i = self.master_rates['Expiry_day_count'][index]/365
            discount = self.master_rates['Discount'][index+1]
            disc_shift = self.master_rates['Discount'][index]
            b = (1/kappa)*(1 - np.exp(-kappa*tau))
            sigma_p = b*vol*np.sqrt((1- np.exp(-2*kappa*t_i))/(2*kappa))
            d1 = (1/sigma_p)*np.log(discount/(disc_shift*strike_input)) + sigma_p/2
            d2 = d1 - sigma_p

            V = (N/strike_input)*(strike_input*disc_shift*norm.cdf(-d2) - discount*norm.cdf(-d1))
            return V

        def cap_black(maturity_index, N, cap_master, vol=False):
            """Returns cap value based on Black formula."""
            maturity = cap_master['Maturity'][maturity_index]
            if vol == False:
                flat_vol = cap_master['Flat_Vol'][maturity_index]
            else:
                flat_vol = vol
            strike = cap_master['ATM Strike'][maturity_index]
            caplet_range = np.arange(1,maturity*4)
            cap_pv = 0
            for index in caplet_range:
                cap_pv += caplet_black(index,N,flat_vol,strike)
            return cap_pv

        def cap_HW(maturity_index, vol, N, cap_master, kappa):
            """Returns cap value based on HW model."""
            maturity = cap_master['Maturity'][maturity_index]
            # flat_vol = cap_master['Flat_Vol'][maturity_index]
            strike = cap_master['ATM Strike'][maturity_index]
            caplet_range = np.arange(1,maturity*4)
            cap_pv = 0
            for index in caplet_range:
                tau = self.master_rates['Tau'][index+1]
                strike_input = (1/(1+strike*tau))
                clet = caplet_hull_white(index,10000000,vol,kappa,strike_input)
                cap_pv += clet

            return cap_pv

        def to_minimize(params):
            kappa = params[0]
            fv = params[1]
            res = 0
            for i in range(15):
                err = cap_black(i, 10000000,  cap_master_df) - cap_HW(i, fv,10000000, cap_master_df, kappa)
                res += err**2
            res = np.sqrt(res)
            return res


        ##Version A: Run this section instead of Version B see minimizer at work - takes a few mins
        #-------------------------------------------------
        # optimum = minimize(to_minimize, x0 = [0.20, 0.010])
        # print("Optimal Kappa: ", optimum.x[0],"\n", "Optimal Vol: ", optimum.x[1])
        # opti_kap = optimum.x[0]
        # opti_vol = optimum.x[1]
        #---------------------------------------------------


        ##Version B: Hardcoded optimized kappa and vol for quicker run-time
        #--------------------------------------------------
        #Uncomment this to run code faster (don't need to run optimizer)
        opti_kap = 0.11469962
        opti_vol = 0.01456547
        #------------------------------------------------------


        cap_price_list_black = []
        cap_price_list_hull = []
        for cap_index in range(len(cap_master_df)):
            maturity = cap_master_df['Maturity'][cap_index]
            flat_vol = cap_master_df['Flat_Vol'][cap_index]
            strike = cap_master_df['ATM Strike'][cap_index]
            caplet_range = np.arange(1,maturity*4)
            caplet_black_pv = []
            caplet_hw_pv = []
            for index in caplet_range:
                caplet_black_pv.append(caplet_black(index,10000000,flat_vol,strike))

                tau = self.master_rates['Tau'][index+1]
                strike_input = (1/(1+strike*tau))
                caplet_hw_pv.append(caplet_hull_white(index,10000000,opti_vol, opti_kap,strike_input))
            cap_price_list_black.append(round(sum(caplet_black_pv),3))
            cap_price_list_hull.append(round(sum(caplet_hw_pv),3))
        cap_master_df['Cap Prices - Black'] = cap_price_list_black
        cap_master_df['Cap Prices - Hull'] = cap_price_list_hull
        if self.show_prints:
            print("\n1d) Summary of Cap Pricing", cap_master_df, " \n Optimized Kappa: ", opti_kap, "\n", "Optimized Vol: ", opti_vol)

        #--------------------------------------------------------------
        # Estimate theta(t) from today t_0 to 30-years with timestep 1/12
        #--------------------------------------------------------------

        #First we need to interpolate the monthly discount factors from quarterly ones

        disc_quart = np.array(self.master_rates ['Discount']).astype(float)
        x_quart = np.array(self.master_rates['T_ACT_360'])
        x_monthly = np.arange(0,30 + (1/12),(1/12))
        disc_monthly_f = interpolate.interp1d(x_quart, disc_quart, kind = 'cubic')
        disc_monthly = disc_monthly_f(x_monthly)

        if self.show_plots:
            plt.plot(x_quart,disc_quart, 'b1', ms = 6, label = 'Quarterly Discount')
            plt.plot(x_monthly,disc_monthly ,'k', label = "Interpolated Monthly Discount")
            #plt.title('e) Interpolated Monthly Forward')
            plt.xlabel('Years')
            plt.ylabel('Discount')
            plt.legend(loc = 'upper right')
            plt.savefig('1e_discount.eps', format='eps')
            plt.show()

        #Calculating Monthly Forwards and derivative using finite differencing
        fm = np.array([-(np.log(disc_monthly[i]) - np.log(disc_monthly[i-1]))/(1/12) for i in range(1,len(disc_monthly))]).astype(float)
        d_fm = np.array([(fm[i] - fm[i-1])/(1/12) for i in range(1,len(fm))]).astype(float)

        self.kappa = opti_kap
        self.sigma = opti_vol

        #Calculating Theta
        self.theta = d_fm + self.kappa*fm[1:] + ((self.sigma**2)/(2*self.kappa))*(1 - np.exp(-2*self.kappa*(x_monthly[2:])))

        if self.show_plots:
            plt.plot(x_monthly[2:], self.theta, 'b')
            #plt.title('Theta vs. Time')
            plt.xlabel('Years')
            plt.ylabel('Theta')
            plt.savefig('1e_theta.eps', format='eps')
            plt.show()

        #--------------------------------------------------------------
        # Converting Hull White prices into Black implied volatilities
        #--------------------------------------------------------------

        #Uncomment to generate HW_fitted_vol element by element by manually changing i.
        #HW_fitted_vol = []

        #def to_solve(vol): #we find the Black volatility such that the Black cap price is equal
        #    #to the Hull White cap price, and we repeat the process for each cap.
        #    i = 0 #we manually update it from 0 to 14
        #    res=np.abs(cap_black( i, 10000000, cap_master_df, self.sigma) - cap_price_list_hull[i])
        #    return res

        #HW_fitted_vol.append(optimize.fsolve(to_solve, 1))
        #for i in range(len(HW_fitted_vol)):
        #    HW_fitted_vol[i] = HW_fitted_vol[i][0]

        HW_fitted_vol = [57.69253532705768, 45.7494830622483, 38.70443281901373, 34.126571197767674, 30.84046463186082, 28.386109815307847, 26.544006407803735, 24.983724899829134, 23.741480325968226, 22.70368753715123, 20.921629763049662, 18.986272863673832, 17.056399816168746, 16.024946526447255, 15.334655293261521]

        initial_vol = list(cap_master_df['Flat_Vol'])

        #plt.plot(np.arange(0,15,1),initial_vol, 'b1', ms = 6, label = 'Black 76 Implied volatility')
        #plt.plot(np.arange(0,15,1),HW_fitted_vol ,'k', label = "1F-HW Calibrated Implied volatility")
        ##plt.title('f) Implied vs actual volatilities')
        #plt.xlabel('Cap index')
        #plt.ylabel('Black Implied Volatility (%)')
        #plt.legend(loc = 'upper right')
        #plt.savefig('1f_vol.eps', format='eps')
        #plt.show()


    def simulate_interest_rates(self, n):
        # The subscript _A denotes that is a tuple containing antithetic paths in each of the two positions
        dt = 1/12
        r0 = self.fi.hull_white_instantaneous_spot_rate(0, 3*dt, self.master_rates.loc[1, 'Discount'], self.theta, self.kappa, self.sigma)
        simulated_rates_A = self.fi.hull_white_simulate_rates_antithetic(n, r0, dt, self.theta, self.kappa, self.sigma)
        simulated_Z_A = self.fi.hull_white_discount_factors_antithetic_path(simulated_rates_A, dt)
        return (simulated_rates_A, simulated_Z_A)

    def calculate_T_year_rate_APR(self, r_A, lag, horizon, previous_rates):
        '''
            Intended to return the 10-year Treasury rate at the end of every pool with a lag of 3 months (horizon is 10).
            Returns a list where in each position (corresponding to each ending month in end) has a tupple with the (antithetic) APR
            r_A contains antithetic simulations of the instantaneous spot rate.
            lag is the lag in months.
            T is the horizon in years of the period to get the APR.
        '''
        n = r_A.shape[0]
        Z_A = self.fi.hull_white_discount_factor(r_A, 0, horizon, self.theta, self.kappa, self.sigma)
        r_APR = 12*((1/Z_A)**(1/(12*horizon)) - 1)
        r_APR[:, lag:] = r_APR[:, :-lag]
        r_APR[:, :lag] = np.array(previous_rates[-lag:])
        return r_APR

