# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:23:41 2020

@author: SUCHARITA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

groceries= [] # as groceries data is transaction data so read it directly as below
with open("F:\\ExcelR\\Assignment\\Association rules\\groceries.csv", "r") as f:
    groceries = f.read()
    groceries= groceries.split("\n") # items bought together are considered as a single transaction 


groceries_list=[]
for i in groceries:
    groceries_list.append(i.split (",")) # help to count how many items purchased in a single transaction
    

#convert to dataframe as "apriori" needs data in form of dataframe and in dummies
    
groceries_series  = pd.DataFrame(pd.Series(groceries_list)) #create dataframe
groceries_series = groceries_series.iloc[:9835,:] 
groceries_series.columns = ["transactions"] # naming the column as transaction
X = groceries_series["transactions"].str.join(sep='*').str.get_dummies(sep='*') # creating dummies
freq_item = apriori(X, min_support = 0.005, max_len = 3, use_colnames = True)

freq_item.sort_values("support", ascending = False, inplace = True)
plt.bar(x= list(range(0,11)),height=freq_item.support[0:11])
plt.xticks(list(range(0,11)),freq_item.itemsets[0:11] )
plt.xlabel("items-sets")
plt.ylabel("support")

rules = association_rules(freq_item, metric="lift", min_threshold=1) # 2700 rules
rules.head(20)
rules.sort_values('lift',ascending = False)


def to_list(i):
    return(sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)    
rules_sets = list(ma_X)
unique = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules= []
for i in unique:
    index_rules.append(rules_sets.index(i))
 
rules_r  = rules.iloc[index_rules,:] # getting rules without any redudancy, 830 rules
rules_r.sort_values('lift',ascending=False).head(10)

plt.scatter(rules_r['support'], rules_r['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')

plt.scatter(rules_r['support'], rules_r['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs lift')

fit= np.polyfit(rules_r['lift'], rules_r['confidence'], 1) # 1 denote degree of polynomial, creating best fit line
fit_fn= np.poly1d(fit)
plt.plot(rules_r['lift'], rules_r['confidence'], 'yo', rules_r['lift'], fit_fn(rules_r['lift']))
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title('lift vs confidence')

#using different support and length value
freq_item1 = apriori(X, min_support = 0.25, max_len = 2, use_colnames = True) 
f1= freq_item1.sort_values("support", ascending = False) # sort the dataset based on support value

plt.bar(x= list(range(0,3)),height=f1.support[0:3])
plt.xticks(list(range(0,3)),freq_item1.itemsets[0:3] )
plt.xlabel("movie-sets")
plt.ylabel("support")

rules1 = association_rules(freq_item1, metric="lift", min_threshold=1) # 2 rules
rules1.shape
rules1.head(20)
rules1.sort_values('lift',ascending = False)

def to_list1(i):
    return(sorted(list(i)))

ma_X1 = rules1.antecedents.apply(to_list1) + rules1.consequents.apply(to_list1)
ma_X1 = ma_X1.apply(sorted)    
rules_sets1 = list(ma_X1)
unique1 = [list(m) for m in set(tuple(i) for i in rules_sets1)] # 0 rule, so need not check redundancy


#using different support and length value
freq_item2 = apriori(X, min_support = 0.02, max_len = 3, use_colnames = True) 
f2= freq_item2.sort_values("support", ascending = False) # sort the dataset based on support value

plt.bar(x= list(range(0,3)),height=f2.support[0:3])
plt.xticks(list(range(0,3)),freq_item2.itemsets[0:3] )
plt.xlabel("movie-sets")
plt.ylabel("support")

rules2 = association_rules(freq_item2, metric="lift", min_threshold=1) # 60 rules
rules2.shape
rules2.head(20)
rules2.sort_values('lift',ascending = False)

def to_list2(i):
    return(sorted(list(i)))

ma_X2 = rules2.antecedents.apply(to_list2) + rules2.consequents.apply(to_list2)
ma_X2 = ma_X2.apply(sorted)    
rules_sets2 = list(ma_X2)
unique2 = [list(m) for m in set(tuple(i) for i in rules_sets2)] #59 rules
index_rules2= []
for i in unique2:
    index_rules2.append(rules_sets2.index(i))
 
rules_r2  = rules2.iloc[index_rules2,:] # getting rules without any redudancy, 59 rules
rules_r2.sort_values('lift',ascending=False).head(10) 

#using different support and length value
freq_item3 = apriori(X, min_support = 0.008, max_len = 5, use_colnames = True) 
f3= freq_item3.sort_values("support", ascending = False) # sort the dataset based on support value

plt.bar(x= list(range(0,3)),height=f3.support[0:3])
plt.xticks(list(range(0,3)),freq_item3.itemsets[0:3] )
plt.xlabel("movie-sets")
plt.ylabel("support")

rules3 = association_rules(freq_item3, metric="lift", min_threshold=1) # 1026 rules
rules3.shape
rules3.head(20)
rules3.sort_values('lift',ascending = False)

def to_list3(i):
    return(sorted(list(i)))

ma_X3 = rules3.antecedents.apply(to_list1) + rules3.consequents.apply(to_list3)
ma_X3 = ma_X3.apply(sorted)    
rules_sets3 = list(ma_X3)
unique3 = [list(m) for m in set(tuple(i) for i in rules_sets3)] # 365 rule, so need for redundancy

index_rules3= []
for i in unique3:
    index_rules3.append(rules_sets3.index(i))
 
rules_r3  = rules3.iloc[index_rules3,:] # getting rules without any redudancy, 365 rules
rules_r3.sort_values('lift',ascending=False).head(10)



























