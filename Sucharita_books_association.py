import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
book= pd.read_csv("F:\\ExcelR\\Assignment\\Association rules\\book.csv")
book.head(5) # already in form of dummy dataframe

freq_book = apriori(book, min_support = 0.005, max_len = 3, use_colnames = True) 
f= freq_book.sort_values("support", ascending = False) # sort the dataset based on support value

plt.bar(x= list(range(0,6)),height=f.support[0:6])
plt.xticks(list(range(0,6)),freq_book.itemsets[0:6] )
plt.xlabel("book-sets")
plt.ylabel("support")

rules = association_rules(freq_book, metric="lift", min_threshold=1) # 1054 rules
rules.shape

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
 
rules_r  = rules.iloc[index_rules,:] # getting rules without any redudancy, 212 rules
rules_r.sort_values('lift',ascending=False).head(10)

# visualization

plt.scatter(rules_r['support'], rules_r['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')

plt.scatter(rules_r['support'], rules_r['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs lift')

fit= np.polyfit(rules_r['lift'], rules_r['confidence'], 1) # 1 denote degree of polynomial
fit_fn= np.poly1d(fit)
plt.plot(rules_r['lift'], rules_r['confidence'], 'gs', rules_r['lift'], fit_fn(rules_r['lift']))
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title('lift vs confidence')

#using different support and length value
freq_item1 = apriori(book, min_support = 0.25, max_len = 2, use_colnames = True) 
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
unique1 = [list(m) for m in set(tuple(i) for i in rules_sets1)] # 1 rule, so need not check redundancy


#using different support and length value
freq_item2 = apriori(book, min_support = 0.02, max_len = 3, use_colnames = True) 
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
unique2 = [list(m) for m in set(tuple(i) for i in rules_sets2)] #139 rules
index_rules2= []
for i in unique2:
    index_rules2.append(rules_sets2.index(i))
 
rules_r2  = rules2.iloc[index_rules2,:] # getting rules without any redudancy, 139 rules
rules_r2.sort_values('lift',ascending=False).head(10) 

#using different support and length value
freq_item3 = apriori(book, min_support = 0.008, max_len = 5, use_colnames = True) 
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
unique3 = [list(m) for m in set(tuple(i) for i in rules_sets3)] # 640 rule, so need for redundancy

index_rules3= []
for i in unique3:
    index_rules3.append(rules_sets3.index(i))
 
rules_r3  = rules3.iloc[index_rules3,:] # getting rules without any redudancy, 640 rules
rules_r3.sort_values('lift',ascending=False).head(10)

