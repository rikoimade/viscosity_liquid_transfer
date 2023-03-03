# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:30:22 2023

@author: riko i made

The following script will try to suggest the optimize parameters to reach 
desired transfer mass

user input:  
    - 'aspiration_rate', 
    - 'dispense_rate', 
    - 'delay_aspirate', 
    - 'delay_dispense',
    
suggestion: ['volume','aspiration_rate', 
         'dispense_rate', 'delay_aspirate', 
         'delay_dispense'] % error
training data: 817

version 2: add blow out rate
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import sklearn
#from sklearn import metrics
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split, LeaveOneOut

from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel

class Squirt:
    
    df = None # prior's dataset
    features = None
    target = None
    model = None # surogate model, either lin = linear, or 'gpr' = gaussian proc
    density = None
    
    asp_max = 25 # aspiration_rate maximum
    asp_min = 20 # aspriation_rate minimum
    
    dsp_max = 13 # dispense_rate maximum
    dsp_min = 8  # dispense_rate minimum
    
    asp_delay_max = 5
    asp_delay_min = 0
    
    dsp_delay_max = 5
    dsp_delay_min = 0
    
    blowout_rate_min = 0
    blowout_rate_max = 10
    
    blowout_delay_min = 0
    blowout_delay_max = 10
    
    vol_min = 100 # micro liter
    vol_max = 1000 # micro liter
    
    def __init__(self, name = 'Unknown'):
        self.name = name
    
    def calibrate(self, volume = list(np.linspace(100,1000,10)), model_kind='gpr'):
        '''
        function to use to calibrate, to find the aspiration and dispense rate
        return: asp_rate, disp_rate
        
        generate surogate function, 
        run gp_minimize to find the next suggestion, with mass constraints
        
        volume_list: insert volume as a list
        '''
        
        if type(volume) != list: volume=[volume]
        
        from warnings import filterwarnings 
        filterwarnings("ignore")
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.df[self.features])
        self.y_train = np.asarray(self.df[self.target])
        
        self.fit(model_kind)
        
        
        self.space = [Categorical(volume, name='volume'),
                      Real(self.asp_min, self.asp_max, name='aspiration_rate'),
                      Real(self.dsp_min, self.asp_max, name='dispense_rate'),
                      Real(self.asp_delay_min, self.asp_delay_max, name='delay_aspirate'),
                      Real(self.dsp_delay_min, self.dsp_delay_max, name='delay_dispense'),
                      Real(self.blowout_rate_min, self.blowout_rate_max, name='blow_out_rate'),
                      Real(self.blowout_delay_min, self.blowout_delay_max, name='delay_blow_out')
                      ]
        
        @use_named_args(self.space)
        def obj_func(**input_array):
            dx = pd.DataFrame()
            for key in input_array.keys():
                
                dx.loc[0,key] = input_array[key] 
                
            # print(dx)
            #input_array = np.asarray(dx)
            
            X = self.scaler.transform(dx)
            
            
            pred = self.model.predict(X)
            
            ## scalarization:
            out = pred.item()/(1/input_array['aspiration_rate'] + 1/input_array['dispense_rate'])
            
            
            return out #pred.item()
        
        self.res = gp_minimize(obj_func, 
                          self.space, 
                          n_calls=50, 
                          kappa = 1.96, # default 1.95 balanced between exploitation vs exploration
                          #x0 = np.asarray(self.df[self.features]),
                          #y0 = self.y_train.reshape(-1,1)
                          )
                          
        self.out_df = pd.DataFrame(data=self.res.x_iters)
        self.out_df.columns = self.features
        self.out_df['%error'] = self.model.predict(self.scaler.transform(self.out_df[self.features]))
        self.out_df['abs-err'] = abs(self.out_df['%error'])
        self.out_df['oo'] = self.out_df['volume']/self.out_df['aspiration_rate'] \
                            + self.out_df['volume']/self.out_df['dispense_rate'] \
                                + self.out_df['delay_aspirate'] + self.out_df['delay_dispense']
        
        self.out_df.sort_values(by='abs-err', inplace=True)
        self.out_df.reset_index(inplace=True, drop=True)
        
        self.out_df2 = self.out_df.iloc[:10,:].copy()
        self.out_df2.sort_values(by='oo', ascending=True, inplace=True)
        self.out_df2.reset_index(inplace=True, drop=True)
        
        
        print('\nNext Run:')
        
        for col in list(self.out_df2)[:-1]:
            print('{:>15}\t: {:.3f}'.format(col, self.out_df2.loc[0,col]))
        #return out_df
    
    def fit(self, kind='gpr'):
        '''
        lin: linear
        gpr: gpr
        '''
        
        if kind == 'gpr':
            matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

            self.model = GaussianProcessRegressor(kernel=matern_tunable, 
                                    n_restarts_optimizer=10, 
                                    alpha=0.5, 
                                      normalize_y=True)
            self.model.fit(self.X_train, self.y_train)
        
        else:
            self.model= sklearn.linear_model.LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            
    
    
    def transfer(self, mass):
        '''
        function to use to transfer liquid in production
        input mass required
        
        return: asp_rate, disp_rate
        '''
        pass
    
    

 
#%%
file_name = 'practice_data.csv'
df = pd.read_csv(file_name)


features = ['volume',
'aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 
'blow_out_rate', 'delay_blow_out']

target='%error'

# %%

liq = Squirt()
liq.name = 'Not-unknown'
liq.density = 0.8466
liq.features = features
liq.df = df
liq.target = target

liq.calibrate() ## input volume, when blank it will chose a value between 100 - 1000 uL, 

df = df.append(liq.out_df2.iloc[0,0:-2],ignore_index=True)
df.iloc[-1,0:5] = df.iloc[0,0:4]
df.loc[:,'touch_tip_aspirate'].iloc[-1] = df.loc[:,'touch_tip_aspirate'].iloc[0]
df.loc[:,'touch_tip_dispense'].iloc[-1] = df.loc[:,'touch_tip_dispense'].iloc[0]
df['m_expected'].iloc[-1]=df['volume'].iloc[-1]/1000 * liq.density
#%%
df['m_measured'].iloc[-1]=  1

df['time'].iloc[-1]=  1 

df[r'%error'].iloc[-1]= (df['m_measured'].iloc[-1]- df['m_expected'].iloc[-1])/df['m_expected'].iloc[-1] *100
df.to_csv('current_experiment.csv', index=False)
#%%

df = pd.read_csv('current_experiment.csv')

#%%

