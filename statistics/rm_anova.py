"""
Name: Ciaran Cooney
Date: 29/12/2018
Description: Program for performing a repeated measures ANOVA on 
4 sets of results -- can be adjusted for different numbers of sets.
"""

import pandas as pd
import numpy as np
from scipy import stats

def calc_grandmean(data, columns):
   """
   Takes a pandas dataframe and calculates the grand mean
   data = dataframe
   columns = list of column names with the response variables
   """
   gm = np.mean(data[columns].mean())
   return gm
   
"""
Sample Data
"""
# X1 = [61.16,70.68,58.59,67.08,59.91,57.78,60.30,60.23,68.83,67.16,62.69,63.38,68.71,60.13,58.75] #svm
# X2 = [70.14,72.12,71.28,72.35,66.67,64.81,69.17,68.18,72.27,71.29,75.63,72.85,72.88,69.76,65] #shallow
# X3 = [66.77,71.29,65.59,69.24,66.34,62.22,49.45,65.91,64.77,70.61,63.51,70.52,77.58,63.7,63.33] #deep
# X4 = [68.86,69.02,69.76,71.59,69.26,62.13,69.09,66.67,65.3,68.56,73.36,70.64,70.61,68.43,67.5] #eegnet

df = pd.DataFrame({'Subid':range(1, len(X1)+1), 'X1':X1, 'X2':X2, 'X3':X3, 'X4':X4})

#####Compute grand mean, decision rule and critical value##### 
grand_mean = calc_grandmean(df, ['X1', 'X2', 'X3', 'X4'])
df['Submean'] = df[['X1', 'X2', 'X3', 'X4']].mean(axis=1)
column_means = df[['X1', 'X2', 'X3', 'X4']].mean(axis=0)

n = len(df['Subid'])
k = len(['X1', 'X2', 'X3', 'X4'])
#Degree of Freedom
ncells = df[['X1','X2','X3', 'X4']].size

dftotal = ncells - 1
dfbw = 3 - 1
dfsbj = len(df['Subid']) -1
dfw = dftotal - dfbw
dferror = dfw - dfsbj
crit_val = stats.f.ppf(q=1-0.05, dfn=dfbw, dfd=dferror)

print(f"Decision rule = dfbetween: {dfbw}, dferror: {dferror}")
print(f"Critical value = {crit_val}")

SSbetween = sum(n*[(m - grand_mean)**2 for m in column_means])

SSwithin = sum(sum([(df[col] - column_means[i])**2 for i,
              col in enumerate(df[['X1', 'X2', 'X3', 'X4']])]))
                                   
SSsubject = sum(k*[(m -grand_mean)**2 for m in df['Submean']])

SSerror = SSwithin - SSsubject
SStotal = SSbetween + SSwithin

"""
print(SSbetween)
print(SSwithin)
print(SSsubject)
print(SSerror)
print(SStotal)
"""

#MSbetween
msbetween = SSbetween/dfbw

#MSerror
mserror = SSerror/dferror

#F-statistic
F = msbetween/mserror

alpha = 0.05
print(msbetween)
print(mserror)
print(f"F-statistic = {F}, for h0 of {crit_val}")
if F > crit_val:
    print(f"Differences are significant with p<{alpha}")
else:
    print("Differences are not significant!")

#Actual p-value
p_value = stats.f.sf(F, 2, dferror)
print(f"p-value = {p_value}")
