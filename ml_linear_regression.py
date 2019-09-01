# data description https://www.kaggle.com/sohier/calcofi#bottle.csv 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#in[1]
import matplotlib.pyplot as plt

from pandas import DataFrame, read_csv
from scipy.stats.stats import pearsonr

### --> read data
#in[2]
bottle = read_csv('bottle.csv', usecols=['Depth_ID', 'Depthm', 'T_degC', 'Salnty'])[1000:2000] # read bottle input
temp = [(int(i[:2]), int(i[3:5]), int(i[5:7]), i[10:12]) for i in bottle['Depth_ID']]
bottle['Century'], bottle['Year'], bottle['Month'], bottle['CastType'] = list(zip(*temp))
bottle = bottle.drop(columns="Depth_ID") 
bottle.fillna(method='ffill', inplace=True)
bottle.dropna(inplace=True)
bottle.head(5)

#in[3]
bottle = bottle[bottle['CastType']=='HY'][bottle['Century']==19][bottle['Year']==49][bottle['Month']==3]
bottle = bottle.drop(columns='CastType')
bottle = bottle.drop(columns='Century')
bottle = bottle.drop(columns='Year')
bottle = bottle.drop(columns='Month')

### --> data description
#in[4]
parameters = ['T_degC', 'Depthm']
objective  = 'Salnty'
bottle.head(5)

#in[5]
x_real = bottle[parameters]
y_real = bottle[objective]

#in[6]
plt.scatter(x_real[parameters[0]], x_real[parameters[1]])
plt.xlabel(parameters[0])
plt.ylabel(parameters[1])

plt.figure()
plt.scatter(x_real[parameters[0]], y_real)
plt.xlabel(parameters[0])
plt.ylabel(objective)

plt.figure()
plt.scatter(x_real[parameters[1]], y_real)
plt.xlabel(parameters[1])
plt.ylabel(objective)

#in[7]
# import sklearn.linear_model.LinearRegression 
from sklearn import linear_model
clf = linear_model.LinearRegression()

# input
X = x_real.values
 
# output
Y = y_real.values

# linear regression model
clf.fit(X, Y)

#in[8]
# He so hoi quy
print('He so=', clf.coef_)

# Sai số
print('Sai so=', clf.intercept_)

# Score
print('Do chinh xac=', clf.score(X, Y))

#in[8]
# sử dụng package matplotlib
import matplotlib.pyplot as plt
 
#in[9]
depth = float(input("Input depth="))
degC  = float(input("Input degC="))
print('predict=', clf.predict([[depth, degC]]))

