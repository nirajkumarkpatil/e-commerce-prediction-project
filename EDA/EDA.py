#WE JOINED THE TWO TABLES BY PIVOTE TABLE.(excel)
#1.HH_DEMOGRAPHIC HAVING HOUSEHOLD AS PRIMARY KEY
#2.TRANSACTION DATA HAVING HOUSEHOLD AS FOREGN KEY(GROUPE BY HOSEHOLD KEY)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as se 
import statsmodels.api as st
data_final=pd.read_csv('C:\\Users\\YC\\Desktop\\pred projectt\\venv\\cleaned_data_hexa.csv')

#DATA NOT CONTAIN ANY NAN VALUE
print(data_final.isnull().value_counts())

#we check the corelation
print(data_final.corr())



#AS WE CHECKED THE CORRELATION QUANTITY AND HOUSE SIZE SHOWS VERY LESS VALUES(-0.003013,0.0022200) SO TO AVOID MULTICOLINEARITY PROBLEM WE DROP THESE TWO COLUMNS
data_final=data_final.drop(['QUANTITY'], axis=1)
data_final=data_final.drop(['house size'], axis=1)


#USED IT FOR SCATTER PLOT TO GET THE IDEA OF LINEARITY
#x=data_final['SALES_VALUE']
#y=data_final['age']
#plt.scatter(x,y)
#plt.show()


#USE TO CHECK NORMAL DITRIBUTION
#TO CHECK THE DATA IS NORMALI DITRIBUTED OR NOT
data_final.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()


#Z SCORE OUTLIERS
# Upper bound
print("Old shape:",data_final.shape)
upper = np.where((data_final['income'].mean)()+3*data_final['income'].std()<data_final['income'])
# Lower bound
lower = np.where((data_final['income'].mean)()-3*data_final['income'].std()>data_final['income'])
#''' Removing the Outliers '''
data_final.drop(upper[0], inplace = True)
data_final.drop(lower[0], inplace = True) 
print("New Shape: ", data_final.shape)
print(data_final.isnull().sum())
#AS WE CHECH OULIERS FOR EACH COLUMN THERE IS NO OULIERS



from sklearn.preprocessing import LabelEncoder,OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn import tree
#from sklearn.neighbors import KNeighborsClassifier, KNeighborsReegressor
#from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error
#from math import sqrt

#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier, KNeighborsReegressor
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import tree


#WE CAN CHECK OUT OUTLIERS BUT NOT A SINGLE COLUMN CONTAIN OUTLIERS
#upper = np.where(pd1['household_key'].mean)()+3*pd1['household_key'].std()<pd1['household_key'])
#lower = np.where(pd1['household_key'].mean)()-3*pd1['household_key'].std()>pd1['household_key']) 
#pd1.drop(upper[0], inplace = True)
#pd1.drop(lower[0], inplace = True) 
#print("New Shape: ", pd1.shape)
#print(pd1.isnull().sum())


X=data_final[['household_key','RETAIL_DISC','COUPON_DISC','COUPON_MATCH_DISC','age','income']]
y=data_final['SALES_VALUE']
X_train, X_test, y_train, y_test= train_test_split (X,y,test_size=0.25)
print(X_train.shape)
print(X_test.shape)

#OUTPUT COLUMN NOT CONTAIN CATOGORICAL VALUES
#le=LabelEncoder()
#y_train=le.fit_transform(y_train)
#y_test=le.fit_transform(y_test)
#print(y_train)
#print(y_test)

#NOT CATOGORICAL COLUMN PRSENT IN THE SELECTED ATTRIBUTES
#oh=OneHotEncoder(drop='first', sparse=False)
#X_train_oh=(oh.fit_transform(X_train['MARITAL_STATUS_CODE']))
#X_test_oh=(oh.fit_transform(X_test['MARITAL_STATUS_CODE']))

#NOT NEED TO ORDINAL BECAUSE NO ANY ORDINAL COLUMN PRESENT
#oe=OrdinalEncoder(categories=['good', 'bad', 'poor'],['dist','first', 'pass', 'fail'])
#X_train=oe.fit_transform(X_train['precipitation'])
#X_test=oe.fit_transform(X_test['precipitation'])


#WE DONE THE STANDARD SCALLING BECAUSE DATA IS NORMALY DISTRIBUTED, 
from sklearn.preprocessing import StandardScaler
stand=StandardScaler()
X_train=stand.fit_transform(X_train)
X_test=stand.fit_transform(X_test)
print(X_train)
print(X_test)


#AMONG ALL THE MODELS LINEAR REGRESSION SHOWS NETTER RESULT WITH EFICIENCY
#THIS IS NOT CASE OF MULTICOLLINEARITY THATS WHY WE AVOID RIDGE LASSO
#THIS IS PROBLEM BASED ON NUMERIC VALUES(CONTINUOUS DATA) THATS WHY WE CANT GO FOR LOGSTIC REGRESSION.
#these values not shows any chance of multicollinearity and no. of attributes also less hence we cant go for the r square and r adjusted values.
#only efficiency and relibility of model is the important.
lr=LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print([lr.predict(X_test)])
print(y_test)
print("EFFICIENCY:",lr.score(X_train, y_train))


#WE FILTER THE DATA FIRST THEN MAKE MODEL BECAUSE OUR DATA IS NOT LIVE. 
#HENCE WE ARE NOT GO FOR PIPELINE


#WEPICKLE FILE FOR DUMPING AND LOADING IN FLASK
import pickle
with open ('my_first_model', 'wb') as swaraj:
    pickle.dump('lr',swaraj) 


with open ('my_first_model', 'rb') as swaraj:
    model=pickle.load(swaraj)

#print(model.predict([[101,100,1.5,3,27,100]]))

#rr = Ridge(alpha=0.01)
#rr.fit(X_train, y_train) 
#pred_train_rr= rr.predict(X_train)
#pred_test_lasso= rr.predict(X_test)
#print(r2_score(y_train, pred_train_rr))
#print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 


#treee=tree.DecisionTreeClassifier()
#treee.fit(X_train, y_train)
#print(treee.score(X_train, y_train))

#model_lasso = Lasso(alpha=0.01)
#model_lasso.fit(X_train, y_train) 
#pred_train_lasso= model_lasso.predict(X_train)
#print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
#print(r2_score(y_train, pred_train_lasso))

#pred_test_lasso= model_lasso.predict(X_test)
#print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
#print(r2_score(y_test, pred_test_lasso))
#print(model_lasso.score(X_train, y_train))


#treee=tree.DecisionTreeClassifier()
#treee.fit(X_train, y_train)
#print(treee.score(X_train, y_train))
