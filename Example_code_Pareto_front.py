# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:34:50 2025

@author: wims
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from paretoset import paretoset

## PREPARE DATA
filename = 'loan_applicants_test_set.csv'
df = pd.read_csv(filename)
df_sub_pop0 = df[df.g == 0]
df_sub_pop1 = df[df.g == 1]
n_tot = df.shape[0]
n_0 = df_sub_pop0.shape[0]
n_1 = df_sub_pop1.shape[0]

## DEFINE PARAMETERS
u00, u01, u10, u11 = 0, 0, -1, 1  	#utility matrix of decision maker
num_thresholds = 100 			#number of thresholds
fairness_metric = 'selection rate'      #options: 'selection rate', 'TPR', 'FPR'  

## DEFINE FUNCTIONS
def get_thresholds(num_thresholds):
    thresholds = np.linspace(0, 1, num_thresholds + 1) 
    return thresholds

def utility_dm(Y, D):			#evaluates utility of decision maker for each individual
    U = (1-abs(Y-D)) * ((1-Y)*u00 + Y*u11) + abs(Y-D) * ((1-Y)*u10 + Y*u01)
    return U

def utility_ds(fairness_metric, Y, D):	#evaluates the fairness metric for each individual
    if(fairness_metric == 'selection rate'):
        V = (1-abs(Y-D)) * Y + abs(Y-D) * (1-Y)
    elif(fairness_metric == 'TPR'):
        V = ((1-abs(Y-D)) * Y) / ((1-abs(Y-D)) * Y + abs(Y-D) * Y)
    elif(fairness_metric == 'FPR'):
        V = (abs(Y-D) * (1-Y)) / ((1-abs(Y-D)) * (1-Y) + abs(Y-D) * (1-Y))
    return V

def evaluate_threshold_lb(t, data):	#evaluates sum of DM utility and the sum of fairness metrics
    dataset = data.copy()
    dataset['D'] = (dataset.p > t).astype(int)
    dataset['U'] = utility_dm(dataset.Y, dataset.D)
    dataset['V'] = utility_ds(fairness_metric, dataset.Y, dataset.D).fillna(0) 
    U_tot = sum(dataset.U)
    V_tot = sum(dataset.V)
    return U_tot, V_tot

## EVALUATE DATA
thresholds = get_thresholds(num_thresholds)

dict_U_0 = {}          
dict_U_1 = {}          
dict_V_0 = {}
dict_V_1 = {}
for t in thresholds:
    U_tot_0, V_tot_0 = evaluate_threshold_lb(t, df_sub_pop0)
    U_tot_1, V_tot_1 = evaluate_threshold_lb(t, df_sub_pop1)
    dict_U_0[t] = U_tot_0
    dict_U_1[t] = U_tot_1
    dict_V_0[t] = V_tot_0
    dict_V_1[t] = V_tot_1

results_combined = []
for t0 in thresholds:
    for t1 in thresholds:
        results_combined.append({
            't0': t0,
            't1': t1,
            'U_tot': dict_U_0[t0] +dict_U_1[t1],
            'U_avg': (dict_U_0[t0] + dict_U_1[t1]) / n_tot,
            'fairness' : abs(dict_V_0[t0]/n_0 - dict_V_1[t1]/n_1)
            })
df_results = pd.DataFrame(results_combined)

## GET PARETO FRONTIER
# mask = paretoset(df_results[['U_avg','fairness']], sense=['max','min'])
# df_paretoset = df_results[mask].sort_values(by='U_avg')

## PLOT RESULTS
plt.scatter(df_results.U_avg, df_results.fairness, s=0.1, color='grey')
# plt.plot(df_paretoset.U_avg, df_paretoset.fairness, color = 'red', label = 'Pareto front')
plt.xlabel('Udm (average per individual)')
plt.ylabel('fairness')
plt.title(fairness_metric)
plt.legend()
# plt.xlim()
# plt.ylim()
plt.show()