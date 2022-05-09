import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as se 
data_final=pd.read_csv('cleaned_data_hexa.csv')
print(data_final)



from sklearn.preprocessing import LabelEncoder,OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import tree
#from sklearn.neighbors import KNeighborsClassifier, KNeighborsReegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV




X=data_final[['household_key','RETAIL_DISC','COUPON_DISC','COUPON_MATCH_DISC','age','income']]
y=data_final['SALES_VALUE']
X_train, X_test, y_train, y_test= train_test_split (X,y,test_size=0.25)
print(X_train.shape)
print(X_test.shape)



from sklearn.preprocessing import StandardScaler
stand=StandardScaler()
X_train=stand.fit_transform(X_train)
X_test=stand.fit_transform(X_test)
print(X_train)
print(X_test)

lr=LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print([lr.predict(X_test)])
print(y_test)
print(lr.score(X_train, y_train))


import pickle
with open ('my__model.pkl', 'wb') as swaraj:
    pickle.dump('lr',swaraj) 


with open ('my__model.pkl', 'rb') as swaraj:
    model=pickle.load(swaraj)


    


