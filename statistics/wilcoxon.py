"""
Name: Ciaran Cooney
Date: 29/12/2018
Description: Wilcoxon signed ranked test.
"""
from scipy.stats import wilcoxon as wlcx
import itertools as it

alpha = 0.05
#n = len(X1)
crit_val = 25 #use table

##### Sample Data #####
# X1 = [70.14,72.12,71.28,72.35,66.67,64.81,69.17,68.18,72.27,71.29,75.63,72.85,72.88,69.76,65]
# X2 = [66.77,71.29,65.59,69.24,66.34,62.22,49.45,65.91,64.77,70.61,63.51,70.52,77.58,63.7,63.33]
# X3 = [68.86,69.02,69.76,71.59,69.26,62.13,69.09,66.67,65.3,68.56,73.36,70.64,70.61,68.43,67.5]
# X4 = [69.37,79.17,72.73,71.52,70.74,63.89,69.92,68.94,72.35,72.12,72.66,71.91,79.02,70.45,67.5]

results_dict = dict(X1=X1, X2=X2, X3=X3, X4=X4)

result_list = list(map(dict, it.combinations(
    results_dict.items(), 2))) #map results list into combinatorial pairs.

#####Iterate through all combinations of results#####
for sub_list in result_list:
    X1 = sub[list(sub_list.keys())[0]]
    X2 = sub[list(sub_list.keys())[1]]
    
    w_stat, p_val = wlcx(X1,X2)
    if w_stat < crit_val:
        print(f"Differences are significant with p<{p_val}")
    else:
        print(f"Differences are not significant! p value = {p_val}")
