import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#STEP 1 IMPORTING DATA
df=pd.read_csv("policy.csv")
df.head()

#FINDING MISSING VALUES
df.apply(lambda x: sum(x.isnull()),axis=0)    #Finding missing values

df['insured_sex'].value_counts()
df['insured_education_level'].value_counts()
df['insured_occupation'].value_counts()
df['insured_relationship'].value_counts()
df['incident_type'].value_counts()
df['incident_severity'].value_counts()
df['authorities_contacted'].value_counts()
df['property_damage'].value_counts()
df['bodily_injuries'].value_counts()
df['police_report_available'].value_counts()
df['fraud_reported'].value_counts()



df['property_damage'] = df['property_damage'].replace({'?':'NO'})
df['property_damage'].value_counts()

df['police_report_available'] = df['police_report_available'].replace({'?':'NO'})
df['police_report_available'].value_counts()

df.dtypes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

var_mod = ['policy_annual_premium','insured_sex','insured_education_level','insured_occupation','insured_relationship','incident_type','incident_severity','authorities_contacted','property_damage','police_report_available','auto_make','fraud_reported']
for i in var_mod:
    df[i]=labelencoder_X.fit_transform(df[i])
df.dtypes


X = df.iloc[:, :-1].values
Y = df.iloc[:, 19].values

#onehotencoder1 = OneHotEncoder(categorical_features = [0])   #DUMMY VAR NEEDED TO BE HANDELED LIKE THIS ONLY
#X = onehotencoder1.fit_transform(X).toarray()

"""
################################################################################################################
#STEP 2 VISUALISING DATA USING PLOT, BARGRAPH AND BOXPLOT
plt.scatter(df['sqft_living15'],df['price'])    #Relation b/w area and price
plt.xlabel('Areas')
plt.ylabel('Price')
plt.show
plt.hist(df['sqft_living15'],bins=50)   #Histogram is used to indicate no of houses of each area category
plt.xlabel('Areas')
plt.ylabel('no of Houses')
plt.show

df.boxplot(column='price',by='sqft_living15')    #Relation b/w area and price

plt.scatter(df['bathrooms'],df['price'])    
plt.xlabel('no of bathrooms')
plt.ylabel('Price')
plt.show
plt.hist(df['bathrooms'],bins=50)   
plt.xlabel('no of bathrooms')
plt.ylabel('no of Houses')
plt.show

df['bathrooms'].value_counts()

df.boxplot(column='price',by='bathrooms')
"""
################################################################################################################
#STEP 3 Applying ML model

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state=None)

#print('Slope: ',  lin_reg.coef_[0])
#print('Intercept: %.3f' % lin_reg.intercept_)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0,max_depth=3)
regressor.fit(X_train, y_train)

################################################################################################################
#STEP 4 Checking for overfitting


y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)


from sklearn.metrics import mean_absolute_error
trainer=mean_absolute_error(y_train, y_train_pred)
tester=mean_absolute_error(y_test, y_test_pred)
print('MSE train: %.3f, test: %.3f' % (trainer,tester))
print(abs(tester-trainer))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_test_pred=y_test_pred.astype(int)
cm = confusion_matrix(y_test, y_test_pred)
