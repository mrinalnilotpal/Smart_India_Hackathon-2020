
# Dated - 7th January 2020


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
%matplotlib inline


df = pd.read_csv("1.10_1.11__rtyb_2006_07.csv")

df.head()

# Now lets have a look at the summary of the data
df.describe()


sdf = df[['Year','Road Transport-Freight Movement (Numbers in Billion Tonnes Kilometres)','Railways-Freight Movement (Numbers in Billion Tonnes Kilometres)']]
sdf.head()




viz = sdf[['Year','Road Transport-Freight Movement (Numbers in Billion Tonnes Kilometres)','Railways-Freight Movement (Numbers in Billion Tonnes Kilometres)']]
viz.hist()
plt.show()


#Now lets plot individually to get the idea of variation year-wise.
# Here I shall rename the function because while plotting it was showing syntax error.

sdf.columns = ['Year' : 'YEAR', 'Road Transport-Freight Movement (Numbers in Billion Tonnes Kilometres)' : 'ROAD']
plt.scatter(sdf.Year, ndf.Road_Transport, color = "blue")
plt.xlabel = ("Year")
plt.ylabel = ("Road_Transport")
plt.show()



# I need to mask and split the data into twp sets. training and split set.
# i am splitting the total data 80% into traininbg set and 20% in to test set.

msk = np.random.rand(len(sdf)) < 0.8
train = sdf[msk]
test = sdf[~msk]



# lets have a look at the trained data distribution.

plt.scatter(train.Year, train.Road_Transport, color = "blue")
plt.xlabel = ("Year")
plt.ylabel = ("Road_Transport")
plt.show()


# modeling of data using sklearn package.

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Year']])
train_y = np.asanyarray(train[['Road_Transport']])
regr.fit(train_x,train_y)

print("Coefficient", regr.coef_)
print("Intercept: ", regr.intercept_)



#plotting the regression.

plt.scatter(train.Year, train.Road_Transport,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Year")
plt.ylabel("Road_Transport")


# Evaluation of model using r_2 score


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


