# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:14:22 2017

@author: rbarnes
"""

from dtw import dtw
import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
import statsmodels.api as sm

df=pd.read_csv('..\\data\\normalized_dtw_data.csv')
print(df)
for var in df.columns.values[1:]:
    df[var]=df[var]/df[var].sum()
print(df)
n=1
distances=[]
ns=[]
week1=[]
week2=[]
for i in range(len(df.columns.values[1:])):
    for y in df.columns.values[1:]:
        x=df.columns.values[i+1]
        dist, cost, acc, path = dtw(df[x], df[y], dist=euclidean)
        distances.append(dist)
        ns.append(n)
        week1.append(x)
        week2.append(y)
        plt.imshow(acc.T, origin='lower', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.xlim((-0.5, acc.shape[0]-0.5))
        plt.ylim((-0.5, acc.shape[1]-0.5))
        plt.title("{0} v. {1}".format(x,y))
        plt.show()

df2=pd.DataFrame()
df2['week1']=week1
df2['week2']=week2
df2['distances']=distances

df2['FEB']=[1 if (obj1=='Sessions_2_5_2017' or obj2=='Sessions_2_5_2016' or obj2=='Sessions_2_1_2015') else 0 for obj1,obj2 in zip(df2['week1'],df2['week2'])]
df2=df2[df2['distances']>0]

print(ttest_ind(df2[df2['FEB']==1]['distances'],df2[df2['FEB']==0]['distances']))

plt.plot(df['Hour Index'],df['Sessions_2_5_2017'],'r')
for var in df.columns.values[2:-2]:
    plt.plot(df['Hour Index'],df[var],'b')
plt.xticks(np.arange(24))
plt.title('Anomalous Sunday (February 5, 2017 in Red)')
plt.xlabel('Hour')
plt.ylabel('Normalized Sessions')
plt.show()


plt.plot(df['Hour Index'],df['Sessions_2_5_2017'],'r')
plt.plot(df['Hour Index'],df['Sessions_2_5_2016'],'b')
plt.plot(df['Hour Index'],df['Sessions_2_1_2015'],'b')
plt.xticks(np.arange(24))
plt.title('Last 3 Super-Bowl Sundays\n(February 5, 2017 in Red)')
plt.xlabel('Hour')
plt.ylabel('Normalized Sessions')
plt.show()

df2['sqrt']=np.sqrt(df2['distances'])

df3=(pd.crosstab(df2['week1'],df2['week2'],np.sqrt(df2['distances']),aggfunc=sum)*pd.crosstab(df2['week1'],df2['week2'],np.sqrt(df2['distances']),aggfunc=sum).T).fillna(0)

df4=pd.DataFrame()
df4['depvar']=[1 if (obj=='Sessions_2_1_2015' or obj=='Sessions_2_5_2016' or obj=='Sessions_2_5_2017') else 0 for obj in df3.index]
df4.index=df3.index
df4['X1']=df3['Sessions_1_29_2017']
df4['X2']=df3['Sessions_1_29_2017']
df4['const']=1
df4.index=df3.index
df4['in']=df4.index

test=df4[(df4.index=='Sessions_2_5_2017')|(df4.index=='Sessions_1_1_2017')]
df4=df4[(df4.index!='Sessions_2_5_2017')|(df4.index!='Sessions_1_1_2017')]

model=sm.Logit(endog=df4['depvar'],exog=df4[['X1','const']])
res=model.fit()


print(model.predict(params=res.params,exog=test[['X1','const']]))
df.to_csv('..\\data\\normalized_dtw_data.csv',index=False)
