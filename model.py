#importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('Iris (1).csv')

x=data.iloc[:,1:5].values
y=data.iloc[:,5].values

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3,random_state=50)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier as RFC
classifier=RFC()
classifier.fit(x_train,y_train)

import pickle
pickle.dump(classifier,open('model.pkl','wb'))